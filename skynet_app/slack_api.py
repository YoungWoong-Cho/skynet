"""Slack settings are selected through the existing workspace middleware."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, SecretStr

from .credential_store import CredentialStoreError
from .slack_notifications import DeliveryError


class SlackConnectRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    webhook_url: SecretStr


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

    @router.post("/connect")
    def connect(request: SlackConnectRequest):
        return checked(
            services.notifications.connect,
            webhook_url=request.webhook_url.get_secret_value(),
        )

    @router.delete("")
    def disconnect():
        return checked(services.notifications.disconnect)

    return router
