# Summarization Toolbox

## Installation

```
pip install -r requirements.txt
```

## Development Instructions

```
pip install pytest
```

Install [fd](https://github.com/sharkdp/fd) and [entr](https://eradman.com/entrproject/) to execute tests automatically on file changes:

```
fd . | entr python -m pytest
```
