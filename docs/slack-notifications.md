# Slack job notifications

Open **Settings → Slack** in your email workspace.

1. [Create a Slack incoming webhook](https://docs.slack.dev/messaging/sending-messages-using-incoming-webhooks/), selecting the channel that should receive your notifications. A private channel works if you are a member. Slack determines the destination through the webhook; Skynet does not change it or configure your Slack push-notification preferences.
2. Paste its URL and select **Connect**. Skynet sends a connection confirmation to Slack before saving the webhook and enabling all five events: submitted, running, cancelled, failed and completed. A failed connection displays an error and remains disconnected.
3. **Disconnect** stops notifications and removes the saved webhook. There is no separate test, enable switch or event configuration.


Your Slack settings and secret webhook are separate from other email workspaces. The webhook is stored in your workspace's secure credential store on the Skynet server, not the browser, SQLite database or job capsule. This uses the same supported secure store as tracking credentials. Email selection remains unverified, as with the rest of Skynet's trusted-team workspace design.

## Events and delivery

Notifications cover training and evaluation workflows, including their individual Slurm attempts. Shared recording/preparation jobs are not assigned to a personal owner and do not produce these notifications. Existing history is not replayed when enabling Slack; future transitions of jobs already running are included.

Submission is reported after Slurm acknowledges a job ID. Cancellation is reported after confirmation, or immediately when Skynet cancels work before submission. A retry's failed attempt is identified as such, with an automatic-retry note. Each new attempt can produce its own submission and start notifications. Successful Slurm exit alone does not mean completion: failed checkpoint or evaluation-result validation produces a failure notification.

Messages contain the experiment name, job type, adapter, Slurm ID, attempt number and event time, plus status/exit code when available. They do not include credentials, full configurations or raw logs. Names are rendered as plain text so they cannot trigger Slack mentions.

State changes and queue entries commit together. A separate worker checks the queue every two seconds; detection of cluster changes follows the existing cluster polling cycle. Delivery continues when the browser is closed. Temporary network errors, Slack server errors and rate limits are retried with backoff, up to six attempts. Slack's Retry-After delay is respected. Settings shows delivery errors. Fix the destination and reconnect if a permanent error stops delivery.

Repeated status polls do not create duplicates. Queue leases prevent concurrent workers from normally sending the same event. Incoming webhooks do not provide exactly-once delivery: if Slack accepts a message but its response is lost, or the server stops before recording the acknowledgement, a retry can deliver a duplicate.

Disconnecting stops notifications, removes the saved webhook and clears that workspace's delivery history. None of these actions cancel jobs. A message already accepted by Slack remains there.

The server requires outbound HTTPS to Slack. It does not require an inbound Slack callback or an OAuth server. The deployment’s optional `SKYNET_PUBLIC_URL` supplies links to the app in notifications; no link is included when it is unset.

## Verification

```bash
python -m pytest tests/test_slack_notifications.py tests/test_email_workspaces.py
npm run test:slack
npm run test:workspaces
```

These tests use actual SQLite transactions, lifecycle methods, the workspace middleware and real settings scripts with mocked Slack delivery. They do not send messages to a real Slack channel or submit cluster jobs.
