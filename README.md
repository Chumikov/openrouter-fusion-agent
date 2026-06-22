# openrouter-fusion-agent

Агент много-модельного обсуждения для [OpenRouter Fusion](https://openrouter.ai/docs/guides/features/server-tools/fusion), работающий **полностью на бесплатных моделях**. Это CLI и [MCP](https://modelcontextprotocol.io)-сервер, который подключается к [opencode](https://opencode.ai) (или любому MCP-клиенту).

OpenRouter Fusion запускает **панель** моделей параллельно (у каждой — веб-поиск), **судья** сравнивает их ответы и возвращает структурированный анализ (консенсус, противоречия, уникальные идеи, слепые зоны), а внешняя модель пишет более сильный финальный ответ. У OpenRouter это по умолчанию дорогие платные модели; проект перенаправляет поток на бесплатные (`:free`) и добавляет **защиту от лимитов free-тиера**.

## Возможности

- 100% бесплатные модели (варианты `:free`) — $0 за запуск.
- Два пресета: **quality** (мощная разнородная панель) и **budget** (мельче/быстрее).
- Budget-aware: читает `GET /api/v1/key`, блокирует при отрицательном балансе (HTTP 402), учитывает дневной лимит 50/1000 и 20 RPM, делает retry на 429.
- MCP-сервер (stdio) с инструментами `fusion_query` и `fusion_status`.
- CLI: разовый запрос + интерактивный REPL.
- mypy strict, pytest, MIT.

## Как это работает

```
вопрос ─▶ внешняя модель ─openrouter:fusion─▶ панель (≤3 free-модели + web_search)
                                                 │
                                                 ▼
                              судья (free-модель + web_search) ─анализ─▶ внешняя модель ─▶ финальный ответ
```

Внешняя модель вызывает серверный инструмент `openrouter:fusion`. Судья возвращает структурированный анализ, который внешняя модель использует для финального ответа. Агент возвращает финальный ответ, стоимость и список участвовавших моделей.

## Установка

Требуется Python ≥ 3.11 и ключ OpenRouter (https://openrouter.ai/keys).

**Без установки** (пакет уже на [PyPI](https://pypi.org/project/openrouter-fusion-agent/)):
```bash
uvx openrouter-fusion-agent "Сравни ridge, lasso и elastic-net регрессию"
```

**Установка:**
```bash
pip install openrouter-fusion-agent          # или: uv tool install openrouter-fusion-agent
```

**Ключ API** нужен только для прогонов fusion (`install-skill` и `print-config` работают без него). Получите ключ на https://openrouter.ai/keys.

Надёжнее держать ключ в файле — это работает и для CLI, и для opencode, и не зависит от оболочки:
```bash
mkdir -p ~/.config/openrouter && umask 077
printf '%s' "sk-or-v1-..." > ~/.config/openrouter/api_key
```
- **opencode (MCP):** `opencode.json` читает ключ из файла через `{file:...}` (см. сниппет ниже) — ничего экспортировать не нужно.
- **CLI:** добавьте в `~/.zshrc` (или `~/.bashrc`) и откройте новый терминал:
```bash
export OPENROUTER_API_KEY="$(cat ~/.config/openrouter/api_key)"
```

> Альтернатива — `{env:OPENROUTER_API_KEY}` в `opencode.json`, если ключ уже экспортирован в оболочке, из которой запускается opencode.

## Использование

**CLI:**
```bash
fusion-agent "Сравни ridge, lasso и elastic-net. Где какая сильнее?"

fusion-agent              # интерактивный REPL
fusion> /status
fusion> /quit
```
Команды REPL: `/status`, `/force on|off`, `/panel 1|2|3`, `/preset quality|budget`, `/budget <n>`, `/help`, `/quit`.
Флаги: `--force on|off` (по умолчанию `on`), `--panel N`, `--preset quality|budget`, `--budget N`.

**opencode (MCP)** — получить готовый блок для `opencode.json`:
```bash
uvx openrouter-fusion-agent print-config
```
или вписать вручную:
```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "fusion": {
      "type": "local",
      "command": ["uvx", "openrouter-fusion-agent", "--mcp"],
      "enabled": true,
      "environment": { "OPENROUTER_API_KEY": "{file:~/.config/openrouter/api_key}" },
      "timeout": 10000
    }
  },
  "experimental": { "mcp_timeout": 90000 }
}
```
После правки конфига перезапустите opencode.

**Скилл (опционально).** Чтобы opencode сам вызывал fusion по подходящим запросам (исследование, сравнение, «аргументы за и против»), установите файл-подсказку:
```bash
uvx openrouter-fusion-agent install-skill            # в текущий проект: .opencode/skills/fusion/
uvx openrouter-fusion-agent install-skill --global   # глобально: ~/.config/opencode/skills/fusion/
uvx openrouter-fusion-agent install-skill --force    # перезаписать существующий
```
Работает на Linux, macOS и Windows (WSL). Для нестандартного расположения конфига opencode задайте `OPENCODE_CONFIG_DIR` — команда учтёт его.

## Пресеты free-моделей

Панель сознательно составлена из разных семейств — ответы меньше коррелируют.

| Пресет | Внешняя (решает + пишет) | Панель (параллельно) | Судья (анализ) |
| --- | --- | --- | --- |
| `quality` (по умолчанию) | `qwen/qwen3-next-80b-a3b-instruct:free` | `openai/gpt-oss-120b:free`, `nvidia/nemotron-3-ultra-550b-a55b:free`, `meta-llama/llama-3.3-70b-instruct:free` | `nvidia/nemotron-3-ultra-550b-a55b:free` |
| `budget` | `qwen/qwen3-next-80b-a3b-instruct:free` | `google/gemma-4-26b-a4b-it:free`, `nvidia/nemotron-3-nano-30b-a3b:free`, `openai/gpt-oss-20b:free` | `nvidia/nemotron-3-super-120b-a12b:free` |

Один прогон ≈ `len(panel) + 2` запросов (5 для панели по умолчанию).

## Лимиты free-тиера

Бесплатные модели OpenRouter: **20 запросов/мин** и дневной лимит **50/день** при балансе **< $10** либо **1000/день** при **≥ $10** (сами вызовы остаются $0). Отрицательный баланс даёт HTTP **402 даже на free-моделях**. Агент читает `GET /api/v1/key`, определяет `is_free_tier` и баланс и соблюдает лимиты, чтобы не падать посреди прогона. Совет: пополнение ≥ $10 снимает дневной лимит до 1000, не делая вызовы платными.

## Лицензия

MIT © Chumikov
