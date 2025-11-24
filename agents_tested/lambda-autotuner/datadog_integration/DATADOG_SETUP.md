# Datadog Setup

## Prerequisites
- Docker installed and running
- Valid Datadog API key

## Quick Start with Docker

Start the Datadog Agent locally:

```bash
docker run -d \
  --name datadog-agent \
  -e DD_API_KEY=<your-full-api-key> \
  -e DD_SITE=datadoghq.com \
  -e DD_APM_ENABLED=true \
  -e DD_LLMOBS_ENABLED=1 \
  -p 8126:8126 \
  -v /var/run/docker.sock:/var/run/docker.sock:ro \
  datadog/agent:latest
```

## Verify Agent is Running

```bash
# Check container status
docker ps | grep datadog-agent

# View agent logs
docker logs datadog-agent
```

## Alternative: Homebrew (macOS)

```bash
brew services start datadog-agent
```

## Running the Agent

Once the Datadog Agent is running, execute the script from the project root:

```bash
# From the project root directory
./datadog_integration/run_agent_with_datadog.sh

# Or cd into the directory first
cd datadog_integration
./run_agent_with_datadog.sh
```

The script will automatically find the `.env` file in either the `datadog_integration/` directory or the parent directory.

## Stop the Agent

```bash
docker stop datadog-agent
docker rm datadog-agent
```
