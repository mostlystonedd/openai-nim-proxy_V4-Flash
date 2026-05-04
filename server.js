const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors({
  origin: '*',
  methods: ['GET', 'POST', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization']
}));
app.options('*', cors());
app.use(express.json());

const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

const SHOW_REASONING = false;
const THINKING_MODE = false;

const MODEL_MAPPING = {
  'gpt-3.5-turbo':  'deepseek-ai/deepseek-v4-flash',
  'gpt-4':          'deepseek-ai/deepseek-v4-flash',
  'gpt-4-turbo':    'deepseek-ai/deepseek-v4-flash',
  'gpt-4o':         'deepseek-ai/deepseek-v4-flash',
  'claude-3-opus':  'deepseek-ai/deepseek-v4-flash',
  'claude-3-sonnet':'deepseek-ai/deepseek-v4-flash',
  'gemini-pro':     'deepseek-ai/deepseek-v4-flash'
};

// Health check
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'OpenAI to NVIDIA NIM Proxy',
    model: 'deepseek-ai/deepseek-v4-flash',
    reasoning_display: SHOW_REASONING,
    thinking_mode: THINKING_MODE ? THINKING_MODE : 'disabled'
  });
});

// List models
app.get('/v1/models', (req, res) => {
  res.json({
    object: 'list',
    data: Object.keys(MODEL_MAPPING).map(id => ({
      id,
      object: 'model',
      created: Date.now(),
      owned_by: 'nvidia-nim-proxy'
    }))
  });
});

// Chat completions
app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;

    const nimModel = MODEL_MAPPING[model] || 'deepseek-ai/deepseek-v4-flash';

    const nimRequest = {
      model: nimModel,
      messages,
      temperature: temperature || 0.6,
      max_tokens: max_tokens || 9024,
      stream: stream || false
    };

    if (THINKING_MODE) {
      nimRequest.extra_body = { chat_template_kwargs: { thinking: THINKING_MODE } };
    }

    const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
      headers: {
        'Authorization': `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json'
      },
      responseType: stream ? 'stream' : 'json'
    });

    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      let buffer = '';
      let reasoningOpen = false;

      response.data.on('data', chunk => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        lines.forEach(line => {
          if (!line.startsWith('data: ')) return;
          if (line.includes('[DONE]')) { res.write(line + '\n'); return; }

          try {
            const data = JSON.parse(line.slice(6));
            const delta = data.choices?.[0]?.delta;
            if (delta) {
              const reasoning = delta.reasoning_content;
              const content = delta.content;
              let out = '';

              if (SHOW_REASONING) {
                if (reasoning && !reasoningOpen) { out = '<think>\n' + reasoning; reasoningOpen = true; }
                else if (reasoning) { out = reasoning; }
                if (content && reasoningOpen) { out += '</think>\n\n' + content; reasoningOpen = false; }
                else if (content) { out += content; }
              } else {
                out = content || '';
              }

              delta.content = out;
              delete delta.reasoning_content;
            }
            res.write(`data: ${JSON.stringify(data)}\n\n`);
          } catch (e) {
            res.write(line + '\n');
          }
        });
      });

      response.data.on('end', () => res.end());
      response.data.on('error', () => res.end());

    } else {
      const choices = response.data.choices.map(choice => {
        let content = choice.message?.content || '';
        if (SHOW_REASONING && choice.message?.reasoning_content) {
          content = '<think>\n' + choice.message.reasoning_content + '\n</think>\n\n' + content;
        }
        return {
          index: choice.index,
          message: { role: choice.message.role, content },
          finish_reason: choice.finish_reason
        };
      });

      res.json({
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model,
        choices,
        usage: response.data.usage || { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
      });
    }

  } catch (err) {
    console.error('Proxy error:', err.message);
    res.status(err.response?.status || 500).json({
      error: {
        message: err.message || 'Internal server error',
        type: 'invalid_request_error',
        code: err.response?.status || 500
      }
    });
  }
});

// 404 catch-all
app.all('*', (req, res) => {
  res.status(404).json({
    error: { message: `Endpoint ${req.path} not found`, type: 'invalid_request_error', code: 404 }
  });
});

// Local dev
if (require.main === module) {
  app.listen(PORT, () => {
    console.log(`Proxy running on port ${PORT}`);
    console.log(`Health: http://localhost:${PORT}/health`);
  });
}

module.exports = app;
