# openrouter-fusion-agent

Агент много-модельного обсуждения для [OpenRouter Fusion](https://openrouter.ai/docs/guides/features/server-tools/fusion), работающий **полностью на бесплатных моделях**. Это CLI и [MCP](https://modelcontextprotocol.io)-сервер, который подключается к [opencode](https://opencode.ai) (или любому MCP-клиенту).

OpenRouter Fusion запускает **панель** моделей параллельно (у каждой — веб-поиск), **судья** сравнивает их ответы и возвращает структурированный анализ (консенсус, противоречия, уникальные идеи, слепые зоны), а внешняя модель пишет более сильный финальный ответ. Для этого проект использует бесплатные модели OpenRouter (`:free`) и добавляет **защиту от лимитов free-тиера** и **автоматическую ротацию моделей** при сбоях.

## Возможности

- 100% бесплатные модели (варианты `:free`) — $0 за запуск.
- **Автоматическая ротация**: при 429/5xx модель заменяется на запасную по приоритету — один сбой не рушит запрос.
- **Обновление моделей**: команда `refresh-models` запрашивает актуальный список free-моделей у OpenRouter и фиксирует отбор в файле.
- Budget-aware: читает `GET /api/v1/key`, блокирует при отрицательном балансе (HTTP 402), учитывает дневной лимит 50/1000 и 20 RPM, делает retry на 429 с учётом `Retry-After`.
- MCP-сервер (stdio) с инструментами `fusion_query`, `fusion_status` и `fusion_refresh_models`.
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

**Ключ API** Получите ключ на https://openrouter.ai/keys.

Надёжнее держать ключ в файле — это работает и для CLI, и для opencode, и не зависит от оболочки:
```bash
mkdir -p ~/.config/openrouter && umask 077
printf '%s' "Ваш ключ sk-or-v1-..." > ~/.config/openrouter/api_key
```
- **opencode (MCP):** `opencode.json` читает ключ из файла через `{file:...}` (см. сниппет ниже) — ничего экспортировать не нужно.

## Использование

**CLI:**
```bash
fusion-agent "Сравни ridge, lasso и elastic-net. Где какая сильнее?"

fusion-agent              # интерактивный REPL
fusion> /status
fusion> /quit
```
Команды REPL: `/status`, `/force on|off`, `/panel 1|2|3`, `/budget <n>`, `/help`, `/quit`.
Флаги: `--force on|off` (по умолчанию `on`), `--panel N`, `--budget N`.

**Обновление моделей** — когда free-модели меняются (новые появляются, старые закрываются):
```bash
fusion-agent refresh-models            # запросить актуальный список и записать в файл
fusion-agent refresh-models --min-b 30 # только модели от 30B
fusion-agent refresh-models --print-only  # показать отбор, не записывая
fusion-agent refresh-models --force    # перезаписать существующий файл
```

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
      "timeout": 300000
    }
  }
}
```
После правки конфига перезапустите opencode. Для обновления списка моделей внутри opencode вызовите MCP-инструмент `fusion_refresh_models`.

**Скилл (опционально).** Чтобы opencode сам вызывал fusion по подходящим запросам (исследование, сравнение, «аргументы за и против»), установите файл-подсказку:
```bash
uvx openrouter-fusion-agent install-skill            # в текущий проект: .opencode/skills/fusion/
uvx openrouter-fusion-agent install-skill --global   # глобально: ~/.config/opencode/skills/fusion/
uvx openrouter-fusion-agent install-skill --force    # перезаписать существующий
```
Работает на Linux, macOS и Windows (WSL). Для нестандартного расположения конфига opencode задайте `OPENCODE_CONFIG_DIR` — команда учтёт его.

## Конфигурация моделей

Модели для каждой роли (внешняя, панель, судья) хранятся **упорядоченными по приоритету** — первый в списке основной, остальные запасные. При сбое (429/5xx) агент автоматически переходит к следующему.

| Роль | Назначение |
| --- | --- |
| outer | Внешняя модель — решает задачу и пишет финальный ответ; прямой вызов. |
| panel | 2-3 модели разных семейств для параллельных ответов. |
| judge | Модель-судья — сравнивает ответы панели и пишет анализ. |

По умолчанию используются встроенные модели. Команда `refresh-models` (или MCP-инструмент `fusion_refresh_models`) запрашивает `GET /api/v1/models`, отбирает free-модели с поддержкой tools, ранжирует по размеру и разнообразию семейств, и записывает результат в `~/.config/openrouter-fusion-agent/models.json`. Файл можно править руками.

Один прогон ≈ `len(panel) + 2` запросов (5 для панели из 3 моделей).

## Лимиты free-тиера

Бесплатные модели OpenRouter: **20 запросов/мин** и дневной лимит **50/день** при балансе **< $10** либо **1000/день** при **≥ $10** (сами вызовы остаются $0). Отрицательный баланс даёт HTTP **402 даже на free-моделях**. Агент читает `GET /api/v1/key`, определяет `is_free_tier` и баланс и соблюдает лимиты, чтобы не падать посреди прогона. Совет: пополнение ≥ $10 снимает дневной лимит до 1000, не делая вызовы платными.

## Лицензия

MIT © Chumikov
