name: Hourly Screener Scan



on:

  schedule:

    # Запуск каждый час (в 00 минут). Можно менять, например '0 */4 * * *' (каждые 4 часа)

    - cron: '0 * * * *'

  workflow_dispatch: # Позволяет запускать кнопку вручную для теста



jobs:

  build:

    runs-on: ubuntu-latest

    steps:

      - name: Checkout code

        uses: actions/checkout@v3



      - name: Set up Python

        uses: actions/setup-python@v4

        with:

          python-version: '3.9'



      - name: Install dependencies

        run: |

          pip install yfinance pandas numpy requests lxml



      - name: Run Screener

        env:

          TG_TOKEN: ${{ secrets.TG_TOKEN }}

          TG_CHAT_ID: ${{ secrets.TG_CHAT_ID }}

        run: python headless_scanner.py



#### Шаг 3: Добавьте Секреты (Токены)

Чтобы не "светить" пароли в коде:

1.  В репозитории GitHub перейдите в **Settings** -> **Secrets and variables** -> **Actions**.

2.  Нажмите **New repository secret**.

3.  Создайте секрет `TG_TOKEN` (вставьте туда токен бота).

4.  Создайте секрет `TG_CHAT_ID` (вставьте ваш ID).



**Готово!** Теперь GitHub сам, без вашего участия, каждый час будет запускать скрипт, проверять рынок и присылать уведомление в Telegram, если найдет новый сигнал.