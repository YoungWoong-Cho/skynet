"""Slack settings are selected through the existing workspace middleware."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from .credential_store import CredentialStoreError
from .slack_notifications import EVENTS, DeliveryError


class SlackSettingsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    webhook_url: SecretStr | None = None
    enabled: bool = True
    events: list[str] = Field(default_factory=lambda: list(EVENTS))
    app_url: str = ""


def slack_router(services):
    router = APIRouter(prefix="/api/notifications/slack", tags=["notifications"])

    def checked(operation, **kwargs):
        try:
            return operation(**kwargs)
        except CredentialStoreError as error:
            raise HTTPException(
                503,
                "The server secure credential store is unavailable. Unlock it and try again.",
            ) from error
        except DeliveryError as error:
            raise HTTPException(502, str(error)) from error
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

    @router.get("")
    def settings():
        return checked(services.notifications.settings)

    @router.put("")
    def save(request: SlackSettingsRequest):
        return checked(
            services.notifications.configure,
            webhook_url=request.webhook_url.get_secret_value()
            if request.webhook_url
            else None,
            enabled=request.enabled,
            events=request.events,
            app_url=request.app_url,
        )

    @router.delete("")
    def disconnect():
        return checked(services.notifications.disconnect)

    @router.post("/test")
    def test():
        return checked(services.notifications.test)

    @router.post("/retry")
    def retry():
        return checked(services.notifications.retry_failed)

    return router
