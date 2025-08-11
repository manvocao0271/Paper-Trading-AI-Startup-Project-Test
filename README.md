# Paper-Trading-AI-Startup-Project

An interactive paper-trading playground featuring:

- Elite Trading GUI (tkinter) in `trading_competition_fixed.py`
- Terminal-based AI vs Human competition in `alpacaPaperTrade.py`
- Real-time mock market data feed and simple AI strategies

## Setup

1) Clone the repo and install dependencies:

```powershell
# From project root
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2) Configure environment variables (do NOT commit secrets):

- Copy `.env.example` to `.env` and fill in your keys
- `.gitignore` ensures `.env` is not tracked

```
ALPACA_API_KEY=your_key
ALPACA_API_SECRET=your_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets
ALPACA_DATA_URL=https://data.alpaca.markets
ALPACA_WEBSOCKET_URL=wss://stream.data.alpaca.markets/v2/iex
```

The project reads keys via environment variables (see `AlpacaConfig` in `alpacaPaperTrade.py`). For convenience, `python-dotenv` is included; add the following to your entry script if not already present:

```python
from dotenv import load_dotenv
load_dotenv()
```

## Run the terminal competition

```powershell
py alpacaPaperTrade.py
```

Controls: Arrow keys to navigate and size orders, B=buy, S=sell, C=close, H=help, Q=quit.

## Run the GUI competition

```powershell
py trading_competition_fixed.py
```

## Security notes

- Never commit real API keys, tokens, or passwords.
- `.env` is ignored by git; share `.env.example` as a template.
- If a key was committed in the past, revoke and rotate it immediately.

## Troubleshooting

- If keyboard input requires admin on Windows, run the terminal as Administrator for the `keyboard` package.
- If tkinter is missing, install a standard Python distribution that includes Tk.