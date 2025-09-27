# Server

Websocket that handles

1. Prompt generation
2. Interpreting code
3. Capabilities (default to whatever for now)

All the user does is register:

1. Tools
2. pLM (take in str, output str)
3. qLM (take in str, output Basemodel)

And the server does the rest
Prompts, run LMs, calls tools, etc.

## User functions:

camelClient = camelServer(plm, qlm, tools)

await camelClient.query(str) -> str
