# EV Competition Checker

A Python tool to automatically scrape, analyze, and notify you about online prize competitions with the best expected value (EV) for entry, focusing on sites like RevComps. The script estimates the true value of prizes, calculates the probability and expected value for each competition, and sends a daily summary email with the best opportunities.

## Features
- **Web Scraping:** Fetches current competitions and details (prize, ticket price, tickets sold, etc.)
- **Prize Value Estimation:** Uses both explicit cash alternatives and smart estimation for non-cash prizes
- **Expected Value Calculation:** Calculates EV for each competition, factoring in accelerating ticket sales as deadlines approach
- **Custom Alerts:** Always includes Land Rover and Camper Van competitions, regardless of EV
- **Email Notifications:** Sends a daily summary to one or more recipients
- **Logging:** Keeps a CSV log of previously seen competitions to avoid duplicate notifications

## Requirements
- Python 3.8+
- See `requirements.txt` for dependencies (requests, beautifulsoup4, pandas, scikit-learn, matplotlib, python-dotenv, etc.)

## Setup
1. **Clone the repository:**
   ```sh
   git clone <your-repo-url>
   cd EV
   ```
2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
3. **Set up environment variables:**
   Create a `.env` file in the project root with the following:
   ```env
   SMTP_USER=your_gmail_address@gmail.com
   SMTP_PASS=your_gmail_app_password
   EMAIL_TO=recipient1@example.com,recipient2@example.com
   ```
   - Use a Gmail App Password (not your main password) for SMTP_PASS.
   - You can specify multiple recipients, separated by commas.

4. **(Optional) Adjust configuration:**
   - Edit `LOG_FILE` and other constants at the top of `EV_Comp_Checker.py` if needed.

## Usage
Run the script manually:
```sh
python EV_Comp_Checker.py
```
Or schedule it (e.g., with Windows Task Scheduler or cron) for daily runs.

## How It Works
- Scrapes the competitions page and each competition's detail page
- Extracts prize, ticket price, tickets sold, total tickets, and cash alternative (if available)
- Estimates cash value for non-cash prizes using keywords and regression
- Calculates expected value (EV) for each comp, factoring in likely last-minute ticket sales
- Sends an email with:
  - All positive-EV competitions ending soon
  - All Land Rover and Camper Van competitions ending within 24 hours
  - The single best EV comp if no positive-EV comps are found

## Environment Variables
- `SMTP_USER`: Gmail address to send emails from
- `SMTP_PASS`: Gmail App Password
- `EMAIL_TO`: Comma-separated list of recipient emails

## Troubleshooting
- **Email not sending?**
  - Make sure you are using a Gmail App Password, not your main password
  - Check your `.env` file for typos
  - Ensure less secure app access is enabled for your Gmail account
- **Scraping issues?**
  - The site layout may have changed; update the scraping logic in `scrape_competition_details`
  - Run the script with a terminal to see debug output
- **No competitions found?**
  - Check your internet connection
  - The competitions page may be empty or blocked

## Contributing
Pull requests and suggestions are welcome! Please open an issue or PR for improvements.

## License
Apache 2.0
