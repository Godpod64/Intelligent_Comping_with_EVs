from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())  
import os
import re
import csv
import smtplib
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse, urljoin
from email.mime.text import MIMEText
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import pickle
import pytz

print("Imports successful!")

import requests
from bs4 import BeautifulSoup
import pandas as pd
from dateutil import parser as dateparser
from scipy.stats import pearsonr
from scipy import stats

# Add this near the top with other constants
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# Add this dictionary near the top after the HEADERS constant
ESTIMATED_PRIZE_VALUES = {
    # Electronics
    'nintendo switch': 300,
    'nintendo switch 2': 350,
    'nintendo switch oled': 350,
    'nintendo switch lite': 200,
    'xbox series x': 500,
    'xbox series s': 250,
    'playstation 5': 500,
    'ps5': 500,
    'iphone 15': 800,
    'iphone 14': 700,
    'iphone 13': 600,
    'samsung galaxy s24': 800,
    'samsung galaxy s23': 700,
    'macbook': 1000,
    'macbook air': 1000,
    'macbook pro': 1500,
    'ipad': 400,
    'ipad pro': 800,
    'airpods': 150,
    'airpods pro': 250,
    'apple watch': 300,
    
    # Dyson products
    'dyson v15': 600,
    'dyson v12': 500,
    'dyson v11': 400,
    'dyson v10': 350,
    'dyson v8': 300,
    'dyson airwrap': 450,
    'dyson supersonic': 350,
    'dyson pure cool': 300,
    'dyson tp00': 300,
    'dyson tp01': 350,
    'dyson tp02': 400,
    'dyson tp04': 500,
    'dyson hp00': 350,
    'dyson hp01': 400,
    'dyson hp04': 550,
    'dyson am07': 300,
    'dyson fan': 300,
    'dyson air purifier': 400,
    'dyson cordless': 400,
    'dyson stick': 400,
    
    # Other appliances
    'shark vacuum': 200,
    'henry vacuum': 150,
    'ninja blender': 100,
    'ninja foodi': 200,
    'air fryer': 100,
    'instant pot': 100,
    'kitchenaid': 300,
    'nespresso': 150,
    'coffee machine': 200,
    
    # TVs
    '55 inch tv': 500,
    '65 inch tv': 800,
    '75 inch tv': 1200,
    'samsung tv': 600,
    'lg tv': 600,
    'sony tv': 700,
    
    # Other
    'bicycle': 500,
    'e-bike': 1000,
    'electric bike': 1000,
    'drone': 300,
    'gopro': 400,
    'camera': 500,
    'laptop': 800,
    'tablet': 300,
    'headphones': 200,
    'smartwatch': 250,
    'speaker': 150,
    'cash': 1,  # For cash prizes, use the actual amount
}

# ─── CONFIG ────────────────────────────────────────────────────────────────────
URL            = "https://www.revcomps.com/current-competitions/"
LOG_FILE       = r"C:\Users\twrgo\EV\ev_log.csv"
TZ             = pytz.timezone("Europe/London")

# SMTP settings (set these in your env for security)
# SMTP settings
SMTP_HOST      = "smtp.gmail.com"
SMTP_PORT      = 587
SMTP_USER      = os.getenv("SMTP_USER")
SMTP_PASS      = os.getenv("SMTP_PASS")
EMAIL_FROM     = SMTP_USER

# Parse EMAIL_TO to support multiple recipients (comma-separated)
EMAIL_TO_RAW   = os.getenv("EMAIL_TO", "")
EMAIL_TO_LIST  = [email.strip() for email in EMAIL_TO_RAW.split(",") if email.strip()]
EMAIL_TO       = EMAIL_TO_LIST[0] if EMAIL_TO_LIST else ""  # Keep first email for backward compatibility
# ────────────────────────────────────────────────────────────────────────────────

CSV_COLUMNS = [
    "Prize", "Comp", "URL",
    "Tickets Available", "Tickets Sold",
    "Cost per ticket", "Cash value of prize",
    "Probability single ticket wins",
    "Total expected gain before cost",
    "Net EV per ticket"
]

# ─── helper ────────────────────────────────────────────────────────────────────
def get_end_datetime_from_timer(soup, tz=TZ):
    """
    Return the naive or tz-aware datetime encoded in the inline
    `birthday = "May 30, 25 22:00:00"` script.  None if not found.
    """
    import re
    from datetime import datetime

    # 1️⃣ locate the <script> that sets `birthday`
    timer_script = soup.find(
        "script",
        string=lambda s: s and "birthday" in s and "countDown" in s
    )
    if not timer_script:
        return None

    # 2️⃣ pull out the quoted value
    m = re.search(r'birthday\s*=\s*["\']([^"\']+)["\']', timer_script.string)
    if not m:
        return None
    raw = m.group(1).strip()                 # e.g. "May 30, 25 22:00:00"

    # 3️⃣ normalise 2-digit year → 4-digit
    raw = re.sub(r',\s*(\d{2})\s', r', 20\1 ', raw)

    # 4️⃣ parse
    dt = datetime.strptime(raw, "%b %d, %Y %H:%M:%S")
    return tz.localize(dt) if dt.tzinfo is None else dt

def calculate_expected_tickets_sold(current_sold, hours_remaining):
    """
    Calculate expected final ticket sales based on accelerating sales pattern.
    Sales increase by 1% per hour starting at 1% for the first hour.
    """
    if hours_remaining <= 0:
        return current_sold
    
    # Cap at 8 hours since we only consider competitions ending within 8 hours
    hours_remaining = min(hours_remaining, 8)
    
    # Calculate cumulative sales increase
    # Hour 1: +1%, Hour 2: +2%, ..., Hour 8: +8%
    cumulative_increase = 0
    for hour in range(1, int(hours_remaining) + 1):
        cumulative_increase += hour  # 1% + 2% + 3% + ... + n%
    
    # Add fractional hour if needed
    if hours_remaining != int(hours_remaining):
        fractional_part = hours_remaining - int(hours_remaining)
        next_hour_rate = int(hours_remaining) + 1
        cumulative_increase += next_hour_rate * fractional_part
    
    # Calculate expected additional sales
    base_sales_rate = current_sold / max(1, 24)  # Assume current sales happened over ~24 hours
    additional_sales = base_sales_rate * (cumulative_increase / 100)
    
    expected_final_sold = current_sold + additional_sales
    return expected_final_sold

def fetch_and_parse(now):
    print("Fetching webpage...")
    res = requests.get(URL)
    res.raise_for_status()
    print("Webpage fetched successfully")
    soup = BeautifulSoup(res.text, "html.parser")
    
    all_competitions = []  # Store all competition data for analysis
    out = []
    all_evs = []  # Track all EV calculations
    user_preference_comps = []  # Track Land Rover and Camper Van competitions
    processed_urls = set()  # Track processed URLs to avoid duplicates

    # each competition is headed by an <h4> with the prize name & link
    h4_elements = soup.find_all("h4")
    print(f"Found {len(h4_elements)} h4 elements")

    for h4 in h4_elements:
        title = h4.get_text(strip=True)
        print(f"Processing: {title}")
        
        # Get competition URL
        link = h4.find('a')
        if not link or not link.get('href'):
            continue
            
        comp_url = urljoin(URL, link['href'])
        
        # Skip if we've already processed this URL (deduplication)
        if comp_url in processed_urls:
            print(f"  -> Skipping duplicate URL: {comp_url}")
            continue
        processed_urls.add(comp_url)
        
        # Scrape detailed information
        details = scrape_competition_details(comp_url)
        if not details:
            continue
            
        # Get data
        sold = details.get('tickets_sold')
        total = details.get('total_tickets')
        cost = details.get('ticket_price', 0.1)
        cash_alt = details.get('cash_alternative')
        end_dt = details.get('draw_date')  # Now this should include timer extraction
        
        if sold is None or total is None:
            continue
            
        print(f"  -> Final data - Sold: {sold}, Total: {total}, Cost: {cost}, Cash: {cash_alt}, End: {end_dt}")

        # Calculate EV for all competitions that have the required data
        if sold is not None and total is not None and cost is not None and cash_alt is not None and sold > 0:
            # Calculate expected tickets sold based on time remaining
            if end_dt:
                time_remaining = end_dt - now
                hours_remaining = time_remaining.total_seconds() / 3600
                expected_sold = calculate_expected_tickets_sold(sold, hours_remaining)
            else:
                expected_sold = sold * 1.05  # Fallback to old logic
                
            N = expected_sold
            p = 1.0 / N
            expected_gain = cash_alt * p
            net_ev = expected_gain - cost
            
            all_evs.append({
                "Prize": title,
                "Net EV": net_ev,
                "URL": comp_url,
                "Cost": cost,
                "Cash": cash_alt,
                "Sold": sold,
                "Expected Sold": expected_sold,
                "Total": total,
                "Hours Remaining": hours_remaining if end_dt else None
            })

        # Check for user preference competitions (Land Rover and Camper Vans) ending within 24 hours
        if end_dt is not None and sold is not None and total is not None and cost is not None:
            # Skip if already ended
            if end_dt <= now:
                continue

            # Calculate time remaining first
            time_remaining = end_dt - now
            hours_remaining = time_remaining.total_seconds() / 3600

            # Only include competitions ending within the next 24 hours for user preferences
            twenty_four_hours_from_now = now + timedelta(hours=24)
            if end_dt <= twenty_four_hours_from_now:
                title_lower = title.lower()
                
                # Check for Land Rover keywords
                land_rover_keywords = ['land rover', 'range rover', 'defender', 'discovery', 'evoque', 'velar', 'sport']
                is_land_rover = any(keyword in title_lower for keyword in land_rover_keywords)
                
                # Check for Camper Van keywords  
                camper_keywords = ['camper', 'motorhome', 'vw california', 'volkswagen california', 'van conversion', 'campervan', 'rv', 'caravan', 't6.1', 't6', 'vw t5', 'vw t4', 'popup', 'pop top', 'poptop', 'van life', 'adventure van']
                is_camper_van = any(keyword in title_lower for keyword in camper_keywords)
                
                # Debug output for user preference detection
                if hours_remaining <= 24:
                    print(f"  -> Checking user preferences for: {title[:50]}")
                    if is_land_rover:
                        print(f"  -> ✅ LAND ROVER DETECTED: {title}")
                    elif is_camper_van:
                        print(f"  -> ✅ CAMPER VAN DETECTED: {title}")
                    elif any(word in title_lower for word in ['land', 'rover', 'range', 'camper', 'van', 'motorhome']):
                        print(f"  -> ⚠️  Partial match (not included): {title}")
                
                if is_land_rover or is_camper_van:
                    # Calculate expected tickets sold and EV even if negative
                    expected_sold = calculate_expected_tickets_sold(sold, hours_remaining)
                    
                    # Calculate EV (even if negative)
                    if cash_alt:
                        N = expected_sold
                        p = 1.0 / N if N > 0 else 0
                        expected_gain = cash_alt * p
                        net_ev = expected_gain - cost
                    else:
                        net_ev = 0  # No cash alternative available
                        expected_gain = 0
                    
                    competition_type = "Land Rover" if is_land_rover else "Camper Van"
                    
                    user_preference_comps.append({
                        "Prize": title,
                        "URL": comp_url,
                        "Cost": cost,
                        "Cash": cash_alt,
                        "Sold": sold,
                        "Expected Sold": expected_sold,
                        "Total": total,
                        "End": end_dt,
                        "Hours Remaining": hours_remaining,
                        "Net EV": net_ev,
                        "Expected Return": expected_gain,
                        "Type": competition_type
                    })
                    
                    print(f"  -> Found {competition_type} competition: {title[:50]} - EV: £{net_ev:.2f}")

        # Check if this competition meets our criteria for positive EV (8-hour window)
        if (sold is not None and total is not None and sold > 0 and 
            cost is not None and cash_alt is not None and end_dt is not None):
            
            # Skip if already ended
            if end_dt <= now:
                continue

            # Only include competitions ending within the next 8 hours (changed from 12)
            eight_hours_from_now = now + timedelta(hours=8)
            if end_dt > eight_hours_from_now:
                continue

            # Calculate expected tickets sold based on time remaining
            time_remaining = end_dt - now
            hours_remaining = time_remaining.total_seconds() / 3600
            expected_sold = calculate_expected_tickets_sold(sold, hours_remaining)

            N = expected_sold
            p = 1.0 / N
            expected_gain = cash_alt * p
            net_ev = expected_gain - cost

            if net_ev > 0:
                out.append({
                    "Prize": title,
                    "URL": comp_url,
                    "Cost": cost,
                    "Cash": cash_alt,
                    "Sold": sold,
                    "Expected Sold": expected_sold,
                    "Total": total,
                    "End": end_dt,
                    "Hours Remaining": hours_remaining,
                    "Net EV": net_ev
                })

    return out, all_evs, user_preference_comps

def load_log():
    if os.path.exists(LOG_FILE):
        return pd.read_csv(LOG_FILE, dtype=str)
    return pd.DataFrame(columns=CSV_COLUMNS)

def save_log(df):
    df.to_csv(LOG_FILE, index=False)

def notify(fresh_competitions, user_preference_comps=None, highest_ev_comp=None):
    try:
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart
        
        # Create email content
        if fresh_competitions:
            subject = f"🎯 Daily Competition Summary - {len(fresh_competitions)} Positive EV Competitions Found!"
            if user_preference_comps:
                subject += f" + {len(user_preference_comps)} Land Rover/Camper Van Competitions"
        else:
            subject = "📊 Daily Competition Summary - No Positive EV Competitions Today"
            if user_preference_comps:
                subject += f" + {len(user_preference_comps)} Land Rover/Camper Van Competitions"
            elif highest_ev_comp:
                subject += f" (Best EV: £{highest_ev_comp['Net EV']:.2f})"
        
        body_lines = [
            "Hi Chris,",
            "",
            "Here is your daily summary of competitions that might be worth entering. We know you love Land Rovers and Camper Vans, so regardless of the calculated odds, this email will always include a section on those competitions anyway, except you'll also get the probability and expected value. You'll get two key metrics for each comp - probability of winning, but also expected value. The expected value is the key one - it calculates odds, balanced with the ticket price. It's only really worth entering a comp if there is a positive EV, so this email will only return competitions that have a high EV (except for the LRs and CVs :-)) Happy Comping!",
            "",
            "📚 Learn more about Expected Value: https://en.wikipedia.org/wiki/Expected_value",
            "",
            "=" * 80,
            "🎯 POSITIVE EV COMPETITIONS (Ending within 8 hours)",
            "=" * 80,
            ""
        ]
        
        if fresh_competitions:
            body_lines.extend([
                f"Found {len(fresh_competitions)} competitions with positive expected value:",
                "",
                "📈 Expected EV calculated using accelerating sales model:",
                "   • Sales increase by 1% per hour as deadline approaches",
                "   • EV based on expected final ticket count, not current count",
                ""
            ])
            
            for i, comp in enumerate(fresh_competitions, 1):
                title = comp.get('Prize', 'Unknown')
                cost = comp.get('Cost', 0)
                # Always use the correct cash value for this comp
                cash_alt = comp.get('Cash') or comp.get('Cash Alternative') or 0
                ev = comp.get('Net EV', 0)
                url = comp.get('URL', '')
                
                # New fields for expected analysis
                current_sold = comp.get('Sold', 0)
                hours_remaining = comp.get('Hours Remaining', 0)
                # Always recalculate expected_sold using the correct function
                expected_sold = calculate_expected_tickets_sold(current_sold, hours_remaining)
                total_tickets = comp.get('Total', 0)
                
                # Calculate additional info
                # Probability: 1 / expected_sold (if > 0)
                probability_pct = (1 / expected_sold * 100) if expected_sold > 0 else 0
                sales_increase_pct = ((expected_sold - current_sold) / max(1, current_sold) * 100) if current_sold > 0 else 0
                
                # Urgency indicator
                if hours_remaining < 2:
                    urgency = "🔥 URGENT"
                elif hours_remaining < 4:
                    urgency = "⚡ HIGH"
                elif hours_remaining < 6:
                    urgency = "⏰ MODERATE"
                else:
                    urgency = "📅 LOW"
                
                body_lines.extend([
                    f"{i}. {title}",
                    f"   ⏰ Time remaining: {hours_remaining:.1f} hours ({urgency} urgency)",
                    f"   💰 Cost: £{cost:.2f}",
                    f"   🏆 Cash Alternative: £{cash_alt:,.0f}" if cash_alt else "   🏆 No cash alternative",
                    f"   🎫 Current: {current_sold:,} sold / Expected final: {expected_sold:,.0f} sold",
                    f"   📈 Expected sales increase: +{sales_increase_pct:.1f}%",
                    f"   📊 Win Probability (expected): {probability_pct:.4f}%",
                    f"   💎 Expected Value: £{ev:.2f}",
                    f"   🔗 {url}",
                    ""
                ])
        else:
            body_lines.extend([
                "❌ No positive EV competitions found ending within the next 8 hours.",
                ""
            ])
            
            # Add the highest EV competition if available
            if highest_ev_comp:
                title = highest_ev_comp.get('Prize', 'Unknown')
                cost = highest_ev_comp.get('Cost', 0)
                cash_alt = highest_ev_comp.get('Cash') or highest_ev_comp.get('Cash Alternative') or 0
                ev = highest_ev_comp.get('Net EV', 0)
                url = highest_ev_comp.get('URL', '')
                
                # New fields for expected analysis
                current_sold = highest_ev_comp.get('Sold', 0)
                hours_remaining = highest_ev_comp.get('Hours Remaining', 0)
                # Always recalculate expected_sold using the correct function
                expected_sold = calculate_expected_tickets_sold(current_sold, hours_remaining)
                total_tickets = highest_ev_comp.get('Total', 0)
                
                # Probability: 1 / expected_sold (if > 0)
                probability_pct = (1 / expected_sold * 100) if expected_sold > 0 else 0
                sales_increase_pct = ((expected_sold - current_sold) / max(1, current_sold) * 100) if current_sold > 0 else 0
                
                # Urgency indicator
                if hours_remaining < 2:
                    urgency = "🔥 URGENT"
                elif hours_remaining < 4:
                    urgency = "⚡ HIGH"
                elif hours_remaining < 6:
                    urgency = "⏰ MODERATE"
                else:
                    urgency = "📅 LOW"
                
                # EV status
                if ev > 0:
                    ev_status = f"✅ Positive: £{ev:.2f}"
                elif ev > -5:
                    ev_status = f"⚠️  Close: £{ev:.2f}"
                else:
                    ev_status = f"❌ Negative: £{ev:.2f}"
                
                body_lines.extend([
                    "📊 HIGHEST EV COMPETITION (ending within 8 hours):",
                    "",
                    f"🥇 {title}",
                    f"   ⏰ Time remaining: {hours_remaining:.1f} hours ({urgency} urgency)",
                    f"   💰 Cost: £{cost:.2f}",
                    f"   🏆 Cash Alternative: £{cash_alt:,.0f}" if cash_alt else "   🏆 No cash alternative",
                    f"   🎫 Current: {current_sold:,} sold / Expected final: {int(expected_sold):,} sold",
                    f"   📈 Expected sales increase: +{sales_increase_pct:.1f}%",
                    f"   📊 Win Probability (expected): {probability_pct:.4f}%",
                    f"   💎 Expected Value: {ev_status}",
                    f"   🔗 {url}",
                    ""
                ])
        
        # Add Land Rover and Camper Van section
        body_lines.extend([
            "=" * 80,
            "🚗 LAND ROVERS AND CAMPER VANS (Ending within 24 hours)",
            "=" * 80,
            ""
        ])
        
        if user_preference_comps:
            body_lines.append(f"Found {len(user_preference_comps)} Land Rover/Camper Van competitions ending within 24 hours:")
            body_lines.append("")
            
            # Sort by hours remaining (most urgent first)
            user_preference_comps_sorted = sorted(user_preference_comps, 
                                                key=lambda x: x.get('Hours Remaining', 999))
            
            for i, comp in enumerate(user_preference_comps_sorted, 1):
                title = comp.get('Prize', 'Unknown')
                cost = comp.get('Cost', 0)
                cash_alt = comp.get('Cash') or comp.get('Cash Alternative') or 0
                ev = comp.get('Net EV', 0)
                url = comp.get('URL', '')
                comp_type = comp.get('Type', 'Unknown')
                
                # Expected analysis
                current_sold = comp.get('Sold', 0)
                hours_remaining = comp.get('Hours Remaining', 0)
                # Always recalculate expected_sold using the correct function
                expected_sold = calculate_expected_tickets_sold(current_sold, hours_remaining)
                total_tickets = comp.get('Total', 0)
                
                # Probability: 1 / expected_sold (if > 0)
                probability_pct = (1 / expected_sold * 100) if expected_sold > 0 else 0
                
                # Urgency indicator
                if hours_remaining < 6:
                    urgency = "🔥 URGENT"
                elif hours_remaining < 12:
                    urgency = "⚡ HIGH"
                elif hours_remaining < 18:
                    urgency = "⏰ MODERATE"
                else:
                    urgency = "📅 LOW"
                
                # EV status
                if ev > 0:
                    ev_status = f"✅ Positive EV: £{ev:.2f}"
                elif ev > -10:
                    ev_status = f"⚠️  Marginal EV: £{ev:.2f}"
                else:
                    ev_status = f"❌ Negative EV: £{ev:.2f}"
                
                body_lines.extend([
                    f"{i}. {comp_type}: {title}",
                    f"   ⏰ Time remaining: {hours_remaining:.1f} hours ({urgency} urgency)",
                    f"   💰 Cost: £{cost:.2f}",
                    f"   🏆 Cash Alternative: £{cash_alt:,.0f}" if cash_alt else "   🏆 No cash alternative",
                    f"   🎫 Current: {current_sold:,} sold / Expected final: {int(expected_sold):,} sold",
                    f"   📊 Win Probability (expected): {probability_pct:.4f}%",
                    f"   💎 {ev_status}",
                    f"   🔗 {url}",
                    ""
                ])
        else:
            body_lines.extend([
                "No Land Rover or Camper Van competitions are running in the next 24 hours.",
                ""
            ])
        
        body_lines.extend([
            "=" * 80,
            "📝 NOTES:",
            "• These calculations assume accelerating ticket sales as deadlines approach",
            "• Positive EV competitions end within 8 hours - act quickly!",
            "• Land Rover & Camper Van competitions shown regardless of EV (within 24 hours)",
            "• Only enter competitions with positive EV for the best value",
            "",
            "Good luck! 🍀"
        ])
        
        body = "\n".join(body_lines)
        
        # Create message
        msg = MIMEMultipart()
        msg['From'] = EMAIL_FROM
        msg['To'] = ", ".join(EMAIL_TO_LIST)  # Display all recipients in To field
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        
        # Send email
        print(f"Connecting to {SMTP_HOST}:{SMTP_PORT}...")
        server = smtplib.SMTP(SMTP_HOST, SMTP_PORT)
        server.starttls()
        print("Logging in...")
        server.login(SMTP_USER, SMTP_PASS)
        print(f"Sending email to {len(EMAIL_TO_LIST)} recipients: {', '.join(EMAIL_TO_LIST)}...")
        text = msg.as_string()
        server.sendmail(EMAIL_FROM, EMAIL_TO_LIST, text)  # Send to all recipients
        server.quit()
        print(f"✅ Email sent successfully to {len(EMAIL_TO_LIST)} recipients!")
        
    except Exception as e:
        print(f"❌ Email sending failed: {e}")
        print("Make sure you have set up Gmail App Password correctly")

def parse_end_date(date_text, now):
    if not date_text:
        return None
    
    # Clean up the text
    date_text = date_text.strip()
    
    # More comprehensive date pattern matching
    patterns = [
        # Current patterns
        r'(\w+day)\s+(\d{1,2})(?:st|nd|rd|th)\s+(\w+)\s+(\d{4})',
        
        # Additional patterns to catch more dates
        r'(\d{1,2})(?:st|nd|rd|th)\s+(\w+)\s+(\d{4})',  # "29th May 2025"
        r'(\w+)\s+(\d{1,2})(?:st|nd|rd|th)\s+(\d{4})',  # "May 29th 2025"
        r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})',          # "29/05/2025" or "29-05-2025"
        r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})',          # "2025/05/29" or "2025-05-29"
        r'Draw[:\s]*(\w+day)\s+(\d{1,2})(?:st|nd|rd|th)\s+(\w+)\s+(\d{4})',  # "Draw: Thursday 29th May 2025"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, date_text, re.IGNORECASE)
        if match:
            try:
                groups = match.groups()
                
                # Handle different pattern formats
                if len(groups) == 4 and groups[0] in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
                    # "Thursday 29th May 2025"
                    day_num, month_name, year = groups[1], groups[2], groups[3]
                elif len(groups) == 3 and groups[0].isdigit():
                    # "29th May 2025" 
                    day_num, month_name, year = groups[0], groups[1], groups[2]
                elif len(groups) == 3 and groups[1].isdigit():
                    # "May 29th 2025"
                    month_name, day_num, year = groups[0], groups[1], groups[2]
                elif len(groups) == 3 and '/' in date_text or '-' in date_text:
                    # Handle numeric dates
                    if groups[0].isdigit() and len(groups[0]) == 4:
                        # "2025/05/29"
                        year, month_num, day_num = groups[0], groups[1], groups[2]
                        parsed_date = datetime(int(year), int(month_num), int(day_num), 23, 59, 59, tzinfo=TZ)
                        return parsed_date
                    else:
                        # "29/05/2025"
                        day_num, month_num, year = groups[0], groups[1], groups[2]
                        parsed_date = datetime(int(year), int(month_num), int(day_num), 23, 59, 59, tzinfo=TZ)
                        return parsed_date
                else:
                    continue
                
                # Parse month name to number
                month_map = {
                    'january': 1, 'february': 2, 'march': 3, 'april': 4,
                    'may': 5, 'june': 6, 'july': 7, 'august': 8,
                    'september': 9, 'october': 10, 'november': 11, 'december': 12,
                    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
                    'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
                }
                
                month_num = month_map.get(month_name.lower())
                if month_num:
                    parsed_date = datetime(int(year), month_num, int(day_num), 23, 59, 59, tzinfo=TZ)
                    return parsed_date
                    
            except (ValueError, IndexError):
                continue
    
    return None

def extract_timer_info(soup):
    print("    -> Looking for timer information in scripts...")
    timer_info = {}
    
    try:
        # Look for JavaScript timer setup
        scripts = soup.find_all('script', string=True)
        print(f"    -> Found {len(scripts)} script tags to check")
        
        for i, script in enumerate(scripts):
            script_content = script.string
            if script_content and 'birthday' in script_content and 'countDown' in script_content:
                print(f"    -> Found timer script in script {i}")
                
                # Extract the birthday date string
                import re
                birthday_match = re.search(r'birthday\s*=\s*["\']([^"\']+)["\']', script_content)
                if birthday_match:
                    birthday_str = birthday_match.group(1)
                    print(f"    -> Found birthday string: {birthday_str}")
                    
                    # Parse the date format "May 30, 25 22:00:00" (Note: 25 likely means 2025)
                    try:
                        # Handle the "25" year format
                        if ', 25 ' in birthday_str:
                            birthday_str = birthday_str.replace(', 25 ', ', 2025 ')
                        
                        end_date = datetime.strptime(birthday_str, "%B %d, %Y %H:%M:%S")
                        end_date = TZ.localize(end_date)
                        
                        timer_info['end_date'] = end_date
                        timer_info['source'] = 'javascript_birthday'
                        print(f"    -> Parsed birthday end date: {end_date}")
                        return timer_info
                        
                    except ValueError as e:
                        print(f"    -> Error parsing birthday date '{birthday_str}': {e}")
        
        # Look for HTML countdown display
        print("    -> Looking for HTML countdown display...")
        countdown_div = soup.find('div', id='countdown')
        if countdown_div:
            print("    -> Found countdown div")
            try:
                days_elem = countdown_div.find('span', id='days')
                hours_elem = countdown_div.find('span', id='hours')
                minutes_elem = countdown_div.find('span', id='minutes')
                seconds_elem = countdown_div.find('span', id='seconds')
                
                if all([days_elem, hours_elem, minutes_elem, seconds_elem]):
                    days = int(days_elem.get_text(strip=True))
                    hours = int(hours_elem.get_text(strip=True))
                    minutes = int(minutes_elem.get_text(strip=True))
                    seconds = int(seconds_elem.get_text(strip=True))
                    
                    print(f"    -> Found countdown: {days}d {hours}h {minutes}m {seconds}s")
                    
                    # Calculate end date from current time + remaining time
                    now = datetime.now(TZ)
                    end_date = now + timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)
                    
                    timer_info['end_date'] = end_date
                    timer_info['source'] = 'countdown_calculation'
                    print(f"    -> Calculated end date: {end_date}")
                    return timer_info
                    
            except (ValueError, AttributeError) as e:
                print(f"    -> Error parsing countdown values: {e}")
        else:
            print("    -> No countdown div found")
        
        print("    -> No timer information found")
        return {}
        
    except Exception as e:
        print(f"    -> Error extracting timer info: {e}")
        return {}

def parse_end_time(timer_info, current_time):
    """Parse timer information to get actual end datetime"""
    try:
        # Method 1: Try to parse end_date if available
        if timer_info.get('end_date'):
            end_date_str = timer_info['end_date']
            
            # Try different date formats
            date_patterns = [
                '%d %B %Y',      # "30 May 2025"
                '%d %b %Y',      # "30 May 2025" (short month)
                '%Y-%m-%d',      # "2025-05-30"
                '%d/%m/%Y',      # "30/05/2025"
                '%d-%m-%Y',      # "30-05-2025"
            ]
            
            for pattern in date_patterns:
                try:
                    # Parse date and assume end of day
                    parsed_date = datetime.strptime(end_date_str.strip(), pattern)
                    # Set to end of day (23:59:59)
                    end_time = parsed_date.replace(hour=23, minute=59, second=59)
                    return TZ.localize(end_time) if end_time.tzinfo is None else end_time
                except ValueError:
                    continue
        
        # Method 2: Try to parse time_remaining if available
        if timer_info.get('time_remaining'):
            time_parts = timer_info['time_remaining']
            if isinstance(time_parts, tuple) and len(time_parts) >= 3:
                try:
                    days = int(time_parts[0]) if len(time_parts) > 2 else 0
                    hours = int(time_parts[1] if len(time_parts) > 1 else time_parts[0])
                    minutes = int(time_parts[2] if len(time_parts) > 2 else time_parts[1])
                    
                    delta = timedelta(days=days, hours=hours, minutes=minutes)
                    return current_time + delta
                except (ValueError, IndexError):
                    pass
        
        # Method 3: Try to parse countdown_element text
        if timer_info.get('countdown_element'):
            countdown_text = timer_info['countdown_element']
            
            # Look for patterns like "2 days 5 hours 30 minutes"
            time_patterns = [
                r'(\d+)\s*days?\s*(\d+)\s*hours?\s*(\d+)\s*min',
                r'(\d+)d\s*(\d+)h\s*(\d+)m',
                r'(\d+)\s*:\s*(\d+)\s*:\s*(\d+)',  # Hours:Minutes:Seconds format
            ]
            
            for pattern in time_patterns:
                match = re.search(pattern, countdown_text, re.IGNORECASE)
                if match:
                    parts = match.groups()
                    try:
                        if len(parts) == 3:
                            # Could be days:hours:minutes or hours:minutes:seconds
                            if ':' in countdown_text:
                                # Assume hours:minutes:seconds
                                hours, minutes, seconds = map(int, parts)
                                delta = timedelta(hours=hours, minutes=minutes, seconds=seconds)
                            else:
                                # Assume days, hours, minutes
                                days, hours, minutes = map(int, parts)
                                delta = timedelta(days=days, hours=hours, minutes=minutes)
                            return current_time + delta
                    except ValueError:
                        continue
        
        return None
        
    except Exception as e:
        print(f"Error parsing end time: {e}")
        return None

def estimate_cash_value_regression(all_competitions):
    """
    Use regression on ticket_price/total_tickets ratio to estimate missing cash values
    """
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LinearRegression
    import matplotlib.pyplot as plt
    
    print("\n📊 CASH VALUE REGRESSION ESTIMATION")
    print("=" * 60)
    
    # Collect competitions with known cash values
    known_data = []
    missing_data = []
    
    for comp in all_competitions:
        if all(field in comp and comp[field] is not None and comp[field] > 0 
               for field in ['Total', 'Cost']):
            
            # Calculate the ratio (our X variable)
            ratio = comp['Cost'] / comp['Total']
            
            # Check if we have a reliable cash value
            cash_value = comp.get('Cash', 0)
            
            # Much stricter validation to exclude suspicious data
            is_reliable = False
            
            if cash_value and cash_value >= 50:
                # Check if this is a high-confidence source
                prize_title = comp.get('Prize', '').lower()
                
                # Gift cards are highly reliable
                if any(word in prize_title for word in ['giftcard', 'egiftcard', 'voucher']) and 'airbnb' not in prize_title:
                    if 50 <= cash_value <= 2000:  # Reasonable gift card range
                        is_reliable = True
                        print(f"  ✅ High confidence (gift card): {comp['Prize'][:40]} - £{cash_value}")
                
                # Explicit "OR £" in title is reliable
                elif ' or £' in prize_title or ' or £' in comp.get('Prize', ''):
                    if cash_value in [1500, 2000, 4000, 7000, 12000, 30000, 55000]:  # Common real cash alternatives
                        is_reliable = True
                        print(f"  ✅ High confidence (explicit OR): {comp['Prize'][:40]} - £{cash_value}")
                
                # Avoid the problematic £6000 and £9000 values unless they're very specific
                elif cash_value in [6000, 9000]:
                    # Only accept if the title strongly suggests this value
                    if '£6000' in comp.get('Prize', '') or '£9000' in comp.get('Prize', ''):
                        is_reliable = True
                        print(f"  ⚠️  Cautious accept: {comp['Prize'][:40]} - £{cash_value}")
                    else:
                        print(f"  ❌ Suspicious £{cash_value}: {comp['Prize'][:40]} - EXCLUDED")
                
                # Other values - be more selective
                elif 100 <= cash_value <= 5000:
                    # Check if it's a round number (more likely to be real)
                    if cash_value % 100 == 0 or cash_value % 500 == 0:
                        is_reliable = True
                        print(f"  ✅ Moderate confidence: {comp['Prize'][:40]} - £{cash_value}")
            
            if is_reliable:
                known_data.append({
                    'Prize': comp['Prize'][:40] + '...' if len(comp['Prize']) > 40 else comp['Prize'],
                    'Ratio': ratio,
                    'Cash_Value': cash_value,
                    'Total_Tickets': comp['Total'],
                    'Ticket_Price': comp['Cost'],
                    'Source': 'reliable'
                })
            else:
                # Competition missing or unreliable cash value
                missing_data.append({
                    'comp_data': comp,
                    'ratio': ratio
                })
    
    if len(known_data) < 3:
        print(f"⚠️  Only {len(known_data)} competitions with reliable cash values - need at least 3 for regression")
        
        # If we don't have enough reliable data, fall back to basic estimation
        print("📊 Using fallback estimation for missing values...")
        for item in missing_data:
            comp = item['comp_data']
            prize_title = comp.get('Prize', '').lower()
            
            # FIRST: Check if this is a gift card and force correct valuation
            gift_card_value = estimate_prize_value(comp.get('Prize', ''))
            if gift_card_value:
                estimated_cash = gift_card_value
                print(f"  🎁 Gift card detected: {comp['Prize'][:45]:<45} | Value: £{estimated_cash}")
                comp['Cash'] = estimated_cash
                comp['cash_estimated'] = True
                continue
            
            # Basic estimates for other items - updated with more realistic values
            if any(word in prize_title for word in ['iphone 16', 'iphone16']):
                estimated_cash = 900  # Latest iPhone
            elif any(word in prize_title for word in ['iphone 15', 'iphone15']):
                estimated_cash = 800
            elif any(word in prize_title for word in ['iphone 14', 'iphone14']):
                estimated_cash = 700
            elif any(word in prize_title for word in ['iphone', 'samsung galaxy s2', 'phone']):
                estimated_cash = 600  # Generic phone
            elif any(word in prize_title for word in ['macbook pro']):
                estimated_cash = 1800
            elif any(word in prize_title for word in ['macbook air', 'macbook']):
                estimated_cash = 1200
            elif any(word in prize_title for word in ['laptop']):
                estimated_cash = 800
            elif any(word in prize_title for word in ['nintendo switch']):
                estimated_cash = 300
            elif any(word in prize_title for word in ['playstation 5', 'ps5']):
                estimated_cash = 450
            elif any(word in prize_title for word in ['xbox series x']):
                estimated_cash = 450
            elif any(word in prize_title for word in ['xbox series s']):
                estimated_cash = 250
            elif any(word in prize_title for word in ['dyson v15']):
                estimated_cash = 600
            elif any(word in prize_title for word in ['dyson v12', 'dyson v11']):
                estimated_cash = 450
            elif any(word in prize_title for word in ['dyson', 'vacuum']):
                estimated_cash = 350
            # Kitchen appliances and gadgets
            elif any(word in prize_title for word in ['ninja foodi', 'ninja blender', 'knife block', 'knife set']):
                estimated_cash = 150  # Kitchen appliances like Ninja Foodi knife blocks
            elif any(word in prize_title for word in ['air fryer', 'instant pot', 'slow cooker']):
                estimated_cash = 120
            elif any(word in prize_title for word in ['kitchenaid', 'stand mixer']):
                estimated_cash = 400
            elif any(word in prize_title for word in ['nespresso', 'coffee machine']):
                estimated_cash = 200
            elif any(word in prize_title for word in ['toaster', 'kettle', 'blender']):
                estimated_cash = 80
            # Watches
            elif any(word in prize_title for word in ['rolex submariner', 'rolex daytona']):
                estimated_cash = 12000  # High-end Rolex
            elif any(word in prize_title for word in ['rolex gmt', 'rolex batman']):
                estimated_cash = 11000
            elif any(word in prize_title for word in ['rolex']):
                estimated_cash = 8000  # Generic Rolex
            elif any(word in prize_title for word in ['watch']) and 'rolex' not in prize_title:
                estimated_cash = 300  # Generic watch
            # Cars
            elif any(word in prize_title for word in ['bmw m3', 'bmw m4', 'audi rs']):
                estimated_cash = 60000  # High-performance cars
            elif any(word in prize_title for word in ['bmw', 'mercedes', 'audi', 'porsche']):
                estimated_cash = 40000  # Premium cars
            elif any(word in prize_title for word in ['car', 'vehicle']):
                estimated_cash = 25000  # Generic car
            # Other categories
            elif any(word in prize_title for word in ['holiday', 'vacation', 'trip']):
                estimated_cash = 3000  # Holiday packages
            elif any(word in prize_title for word in ['gold', 'bullion']):
                estimated_cash = 2000  # Gold items
            elif any(word in prize_title for word in ['bbq', 'grill']):
                estimated_cash = 800  # BBQ equipment
            elif any(word in prize_title for word in ['drone']):
                estimated_cash = 600
            elif any(word in prize_title for word in ['tent', 'camping']):
                estimated_cash = 500
            elif any(word in prize_title for word in ['tools', 'tool']):
                estimated_cash = 400
            else:
                estimated_cash = 200  # Conservative default (reduced from 300)
            
            comp['Cash'] = estimated_cash
            comp['cash_estimated'] = True
            print(f"  💡 Basic estimate: {comp['Prize'][:45]:<45} | Est: £{estimated_cash}")
        
        return all_competitions
    
    # Create DataFrame
    df = pd.DataFrame(known_data)
    
    print(f"\n📈 High-Quality Training Data ({len(df)} competitions):")
    print("-" * 90)
    print(f"{'Prize':<40} {'Ratio':<10} {'Cash':<10} {'Total':<8} {'Price':<8}")
    print("-" * 90)
    
    for _, row in df.iterrows():
        print(f"{row['Prize']:<40} {row['Ratio']:<10.6f} £{row['Cash_Value']:<9.0f} {row['Total_Tickets']:<8.0f} £{row['Ticket_Price']:<7.2f}")
    
    # Prepare data for regression
    X = df[['Ratio']].values
    y = df['Cash_Value'].values
    
    # Fit regression model
    model = LinearRegression()
    model.fit(X, y)
    
    # Calculate R-squared
    r_squared = model.score(X, y)
    slope = model.coef_[0]
    intercept = model.intercept_
    
    print(f"\n📏 REGRESSION RESULTS:")
    print(f"Equation: Cash_Value = {slope:.2f} × Ratio + {intercept:.2f}")
    print(f"R-squared: {r_squared:.4f}")
    print(f"Correlation strength: {'Strong' if r_squared > 0.7 else 'Moderate' if r_squared > 0.3 else 'Weak'}")
    
    # Only use regression if correlation is decent
    if missing_data and r_squared > 0.1:  # Lower threshold for initial testing
        print(f"\n💰 ESTIMATING CASH VALUES FOR {len(missing_data)} COMPETITIONS:")
        print("-" * 80)
        
        for item in missing_data:
            comp = item['comp_data']
            ratio = item['ratio']
            
            # Predict cash value
            estimated_cash = model.predict([[ratio]])[0]
            
            # Apply reasonable bounds and some heuristics
            prize_title = comp.get('Prize', '').lower()
            
            # Use title-based bounds
            if any(word in prize_title for word in ['nintendo', 'switch', 'playstation', 'xbox']):
                estimated_cash = max(200, min(estimated_cash, 600))
            elif any(word in prize_title for word in ['iphone', 'samsung', 'phone']):
                estimated_cash = max(400, min(estimated_cash, 1200))
            elif any(word in prize_title for word in ['laptop', 'macbook']):
                estimated_cash = max(600, min(estimated_cash, 2000))
            elif any(word in prize_title for word in ['rolex', 'watch']):
                estimated_cash = max(300, min(estimated_cash, 10000))
            elif any(word in prize_title for word in ['ninja foodi', 'knife block', 'knife set', 'air fryer', 'kitchen']):
                estimated_cash = max(50, min(estimated_cash, 300))  # Kitchen appliances
            elif any(word in prize_title for word in ['bbq', 'grill']):
                estimated_cash = max(100, min(estimated_cash, 1000))
            elif any(word in prize_title for word in ['tv', 'television', 'samsung tv', 'lg tv']):
                estimated_cash = max(200, min(estimated_cash, 1500))
            else:
                estimated_cash = max(100, min(estimated_cash, 2000))  # General bounds
            
            # Update the competition data
            comp['Cash'] = estimated_cash
            comp['cash_estimated'] = True
            
            print(f"{comp['Prize'][:45]:<45} | Ratio: {ratio:6.4f} | Est. Cash: £{estimated_cash:,.0f}")
    else:
        print(f"\n⚠️  Using basic estimation (R² = {r_squared:.3f} too weak for regression)")
        
        # Fall back to basic keyword-based estimation
        for item in missing_data:
            comp = item['comp_data']
            prize_title = comp.get('Prize', '').lower()
            
            # FIRST: Check if this is a gift card and force correct valuation
            gift_card_value = estimate_prize_value(comp.get('Prize', ''))
            if gift_card_value:
                estimated_cash = gift_card_value
                print(f"  🎁 Gift card detected: {comp['Prize'][:45]:<45} | Value: £{estimated_cash}")
                comp['Cash'] = estimated_cash
                comp['cash_estimated'] = True
                continue
            
            # Basic estimates for other items - updated with more realistic values
            if any(word in prize_title for word in ['iphone 16', 'iphone16']):
                estimated_cash = 900  # Latest iPhone
            elif any(word in prize_title for word in ['iphone 15', 'iphone15']):
                estimated_cash = 800
            elif any(word in prize_title for word in ['iphone 14', 'iphone14']):
                estimated_cash = 700
            elif any(word in prize_title for word in ['iphone', 'samsung galaxy s2', 'phone']):
                estimated_cash = 600  # Generic phone
            elif any(word in prize_title for word in ['macbook pro']):
                estimated_cash = 1800
            elif any(word in prize_title for word in ['macbook air', 'macbook']):
                estimated_cash = 1200
            elif any(word in prize_title for word in ['laptop']):
                estimated_cash = 800
            elif any(word in prize_title for word in ['nintendo switch']):
                estimated_cash = 300
            elif any(word in prize_title for word in ['playstation 5', 'ps5']):
                estimated_cash = 450
            elif any(word in prize_title for word in ['xbox series x']):
                estimated_cash = 450
            elif any(word in prize_title for word in ['xbox series s']):
                estimated_cash = 250
            elif any(word in prize_title for word in ['dyson v15']):
                estimated_cash = 600
            elif any(word in prize_title for word in ['dyson v12', 'dyson v11']):
                estimated_cash = 450
            elif any(word in prize_title for word in ['dyson', 'vacuum']):
                estimated_cash = 350
            # Kitchen appliances and gadgets
            elif any(word in prize_title for word in ['ninja foodi', 'ninja blender', 'knife block', 'knife set']):
                estimated_cash = 150  # Kitchen appliances like Ninja Foodi knife blocks
            elif any(word in prize_title for word in ['air fryer', 'instant pot', 'slow cooker']):
                estimated_cash = 120
            elif any(word in prize_title for word in ['kitchenaid', 'stand mixer']):
                estimated_cash = 400
            elif any(word in prize_title for word in ['nespresso', 'coffee machine']):
                estimated_cash = 200
            elif any(word in prize_title for word in ['toaster', 'kettle', 'blender']):
                estimated_cash = 80
            # Watches
            elif any(word in prize_title for word in ['rolex submariner', 'rolex daytona']):
                estimated_cash = 12000  # High-end Rolex
            elif any(word in prize_title for word in ['rolex gmt', 'rolex batman']):
                estimated_cash = 11000
            elif any(word in prize_title for word in ['rolex']):
                estimated_cash = 8000  # Generic Rolex
            elif any(word in prize_title for word in ['watch']) and 'rolex' not in prize_title:
                estimated_cash = 300  # Generic watch
            # Cars
            elif any(word in prize_title for word in ['bmw m3', 'bmw m4', 'audi rs']):
                estimated_cash = 60000  # High-performance cars
            elif any(word in prize_title for word in ['bmw', 'mercedes', 'audi', 'porsche']):
                estimated_cash = 40000  # Premium cars
            elif any(word in prize_title for word in ['car', 'vehicle']):
                estimated_cash = 25000  # Generic car
            # Other categories
            elif any(word in prize_title for word in ['holiday', 'vacation', 'trip']):
                estimated_cash = 3000  # Holiday packages
            elif any(word in prize_title for word in ['gold', 'bullion']):
                estimated_cash = 2000  # Gold items
            elif any(word in prize_title for word in ['bbq', 'grill']):
                estimated_cash = 800  # BBQ equipment
            elif any(word in prize_title for word in ['drone']):
                estimated_cash = 600
            elif any(word in prize_title for word in ['tent', 'camping']):
                estimated_cash = 500
            elif any(word in prize_title for word in ['tools', 'tool']):
                estimated_cash = 400
            else:
                estimated_cash = 200  # Conservative default (reduced from 300)
            
            comp['Cash'] = estimated_cash
            comp['cash_estimated'] = True
            print(f"  💡 Basic estimate: {comp['Prize'][:45]:<45} | Est: £{estimated_cash}")
    
    return all_competitions

def estimate_prize_value(prize_title):
    """Enhanced prize value estimation with better gift card and product detection"""
    title_lower = prize_title.lower()
    
    # PRIORITY 1: Handle gift cards first - these should always take precedence
    gift_card_patterns = [
        r'£(\d+(?:,\d{3})*)\s*(?:amazon|currys|sainsburys|one4all|airbnb|tui|voucher|giftcard|egiftcard)',
        r'(\d+(?:,\d{3})*)\s*(?:amazon|currys|sainsburys|one4all|airbnb|tui|voucher|giftcard|egiftcard)',
    ]
    
    for pattern in gift_card_patterns:
        match = re.search(pattern, title_lower)
        if match:
            try:
                amount = float(match.group(1).replace(',', ''))
                if 10 <= amount <= 5000:  # Reasonable gift card range
                    print(f"    -> Detected gift card value: £{amount}")
                    return amount
            except ValueError:
                continue
    
    # PRIORITY 2: Look for explicit cash amounts in the title
    cash_patterns = [
        r'or\s*£([\d,]+)(?:\s|$)',  # "or £5000" at end or followed by space
        r'£([\d,]+)\s*cash',
        r'cash\s*£([\d,]+)', 
        r'([\d,]+)\s*pounds?\s*cash',
        r'cash\s*prize\s*£([\d,]+)',
    ]
    
    for pattern in cash_patterns:
        match = re.search(pattern, title_lower)
        if match:
            try:
                cash_value = float(match.group(1).replace(',', ''))
                if 100 <= cash_value <= 300000:  # Reasonable cash prize range
                    print(f"    -> Detected cash alternative: £{cash_value}")
                    return cash_value
            except ValueError:
                continue
    
    # PRIORITY 3: Return None for manual estimation - don't guess randomly
    print(f"    -> No reliable value detected for: {prize_title[:50]}")
    return None

def scrape_competition_details(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        # 1. Get prize title
        prize_title = ""
        title_element = soup.find('h1', class_='product_title')
        if title_element:
            prize_title = title_element.get_text().strip()

        # 2. Look for cash alternative in the main product summary/description
        description_blocks = [
            soup.find('div', class_='product-summary'),
            soup.find('div', class_='woocommerce-product-details__short-description'),
            soup.find('div', class_='elementor-widget-container'),
            soup.find('div', class_='product-info'),
            soup.find('div', id='product-details'),
        ]
        cash_alternative = None
        for block in description_blocks:
            if block:
                text = block.get_text(separator=' ', strip=True)
                match = re.search(r'(?:OR|or)\s*£([\d,]+)', text)
                if match:
                    cash_alternative = float(match.group(1).replace(',', ''))
                    print(f"  -> Found cash alternative in description: £{cash_alternative}")
                    break

        # 3. If not found, look for explicit cash alternative in the title
        if not cash_alternative and prize_title:
            match = re.search(r'(?:OR|or)\s*£([\d,]+)', prize_title)
            if match:
                cash_alternative = float(match.group(1).replace(',', ''))
                print(f"  -> Found cash alternative in title: £{cash_alternative}")

        # 4. If still not found, fallback to estimation
        if not cash_alternative:
            cash_alternative = estimate_prize_value(prize_title)
            if cash_alternative:
                print(f"  -> Estimated cash alternative: £{cash_alternative}")

        # ... (rest of your scraping logic for tickets, price, etc.)

        return {
            # ... (other fields)
            'cash_alternative': cash_alternative,
            'prize_title': prize_title,
            # ...
        }
    except Exception as e:
        print(f"  -> Error scraping {url}: {e}")
        return {}

def analyze_cash_correlation_and_estimate(all_competitions):
    """Analyze correlation between ticket metrics and cash values, then estimate missing values"""
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import stats
    
    print("\n🔬 CASH VALUE CORRELATION ANALYSIS")
    print("=" * 60)
    
    # Collect competitions with HIGH-CONFIDENCE cash values only
    known_cash_data = []
    missing_cash_comps = []
    
    for comp in all_competitions:
        # Check if we have the required fields
        if all(field in comp and comp[field] is not None and comp[field] > 0 
               for field in ['Total', 'Cost', 'Cash']):
            
            cash_value = comp['Cash']
            
            # Only include if cash value seems realistic and accurate
            # Filter out clearly wrong values (likely parsing errors)
            if (cash_value >= 100 and  # At least £100 (anything less is probably wrong)
                cash_value <= 300000 and  # At most £300k (reasonable upper bound)
                cash_value not in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 25, 30]):  # Common parsing errors
                
                # Additional check: if cash value is suspiciously round, it's more likely to be real
                # Real cash alternatives are often round numbers like £1000, £5000, £20000, etc.
                is_round = (cash_value % 100 == 0) or (cash_value % 500 == 0) or (cash_value % 1000 == 0)
                
                # Or if the prize title contains "OR £" which indicates a clear cash alternative
                has_or_pattern = 'OR £' in comp['Prize'].upper() or 'or £' in comp['Prize']
                
                if is_round or has_or_pattern or cash_value >= 1000:
                    # Calculate Column E: (ticket_price / total_tickets) * 10,000
                    column_e = (comp['Cost'] / comp['Total']) * 10000
                    
                    known_cash_data.append({
                        'Prize': comp['Prize'][:50] + '...' if len(comp['Prize']) > 50 else comp['Prize'],
                        'Total_Tickets': comp['Total'],
                        'Ticket_Price': comp['Cost'], 
                        'Cash_Value': cash_value,
                        'Metric_E': column_e
                    })
                else:
                    print(f"⚠️  Excluding suspicious cash value: {comp['Prize'][:40]} - £{cash_value}")
            else:
                print(f"⚠️  Excluding unrealistic cash value: {comp['Prize'][:40]} - £{cash_value}")
                
        elif all(field in comp and comp[field] is not None and comp[field] > 0 
                 for field in ['Total', 'Cost']) and ('Cash' not in comp or comp['Cash'] == 0 or comp['Cash'] < 100):
            # Competition missing cash value or has unrealistic cash value
            missing_cash_comps.append(comp)
    
    if len(known_cash_data) < 3:
        print(f"⚠️  Only {len(known_cash_data)} competitions with reliable cash values - need at least 3 for analysis")
        return None
        
    # Create DataFrame
    df = pd.DataFrame(known_cash_data)
    
    print(f"📊 Analysis Table ({len(df)} competitions with HIGH-CONFIDENCE cash values):")
    print("-" * 100)
    print(f"{'Prize':<35} {'Total':<8} {'Price':<8} {'Cash':<10} {'Metric E':<10}")
    print("-" * 100)
    
    for _, row in df.iterrows():
        print(f"{row['Prize']:<35} {row['Total_Tickets']:<8.0f} £{row['Ticket_Price']:<7.2f} £{row['Cash_Value']:<9.0f} {row['Metric_E']:<10.4f}")
    
    # Calculate correlation
    correlation, p_value = stats.pearsonr(df['Metric_E'], df['Cash_Value'])
    print(f"\n📈 CORRELATION ANALYSIS:")
    print(f"Correlation coefficient: {correlation:.4f}")
    print(f"P-value: {p_value:.6f}")
    print(f"Correlation strength: {'Strong' if abs(correlation) > 0.7 else 'Moderate' if abs(correlation) > 0.4 else 'Weak'}")
    
    # Only proceed with regression if we have reasonable correlation
    if abs(correlation) > 0.3:  # Only if correlation is at least moderate
        # Calculate line equation using linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(df['Metric_E'], df['Cash_Value'])
        
        print(f"\n📏 LINE EQUATION:")
        print(f"Cash Value = {slope:.2f} × Metric_E + {intercept:.2f}")
        print(f"R-squared: {r_value**2:.4f}")
        print(f"Standard error: {std_err:.2f}")
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        plt.scatter(df['Metric_E'], df['Cash_Value'], alpha=0.7, s=60, color='blue')
        
        # Plot the regression line
        x_line = np.linspace(df['Metric_E'].min(), df['Metric_E'].max(), 100)
        y_line = slope * x_line + intercept
        plt.plot(x_line, y_line, 'r-', linewidth=2, label=f'y = {slope:.2f}x + {intercept:.2f}')
        
        plt.xlabel('Metric E: (Ticket Price / Total Tickets) × 10,000')
        plt.ylabel('Cash Value (£)')
        plt.title(f'Cash Value vs Ticket Metrics (High-Confidence Data Only)\nCorrelation: {correlation:.3f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add some annotations for interesting points
        for i, row in df.iterrows():
            if row['Cash_Value'] > df['Cash_Value'].quantile(0.8):  # Top 20% of cash values
                plt.annotate(row['Prize'][:20], 
                            (row['Metric_E'], row['Cash_Value']),
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=8, alpha=0.7)
        
        plt.tight_layout()
        plt.show()
        
        # Now estimate cash values for missing competitions
        if missing_cash_comps and abs(correlation) > 0.5:  # Only estimate if correlation is decent
            print(f"\n💰 ESTIMATING CASH VALUES FOR {len(missing_cash_comps)} COMPETITIONS:")
            print("-" * 80)
            
            for comp in missing_cash_comps:
                metric_e = (comp['Cost'] / comp['Total']) * 10000
                estimated_cash = slope * metric_e + intercept
                
                # Apply some bounds (cash value shouldn't be negative or unreasonably high)
                estimated_cash = max(100, min(estimated_cash, 100000))  # Between £100 and £100k
                
                comp['Cash'] = estimated_cash
                comp['cash_estimated'] = True
                
                print(f"{comp['Prize'][:45]:<45} | Metric E: {metric_e:6.2f} | Est. Cash: £{estimated_cash:,.0f}")
        else:
            print(f"\n⚠️  Correlation too weak ({correlation:.3f}) - not estimating cash values")
    else:
        print(f"\n⚠️  Correlation too weak ({correlation:.3f}) - skipping regression analysis")
    
    return correlation if len(known_cash_data) >= 3 else None

def calculate_all_evs(competitions):
    """Calculate expected value for ALL competitions (including negative EVs)"""
    print("📊 Calculating expected values for all competitions...")
    
    all_ev_comps = []
    
    for comp in competitions:
        # Check required fields
        required_fields = ['Total', 'Sold', 'Cost']
        missing_fields = [field for field in required_fields if field not in comp or comp[field] is None]
        
        if missing_fields:
            print(f"🔍 Debug - Competition '{comp.get('Prize', 'Unknown')}' has fields: {list(comp.keys())}")
            print(f"⚠️  Skipping '{comp.get('Prize', 'Unknown')}' - missing fields: {missing_fields}")
            continue
        
        # Handle both 'Cash' and 'cash_alternative' field names
        cash_value = comp.get('Cash') or comp.get('cash_alternative') or comp.get('Cash Alternative')
        
        if not cash_value or cash_value <= 0:
            print(f"⚠️  Skipping '{comp.get('Prize', 'Unknown')}' - no valid cash value: {cash_value}")
            continue
        
        try:
            # Use the correct field names
            total_tickets = float(comp['Total'])
            current_sold = float(comp['Sold']) 
            expected_sold = float(comp.get('Expected Sold', current_sold))  # Use expected if available
            ticket_cost = float(comp['Cost'])
            cash_value = float(cash_value)
            
            # Skip if any values are invalid
            if total_tickets <= 0 or ticket_cost <= 0 or cash_value <= 0:
                continue
            
            # Calculate probability of winning based on expected ticket sales
            remaining_tickets = total_tickets - expected_sold
            if remaining_tickets <= 0:
                print(f"⚠️  Skipping '{comp.get('Prize', 'Unknown')}' - expected to sell out (expected sold: {expected_sold}, total: {total_tickets})")
                continue
                
            prob_win = 1 / remaining_tickets
            
            # Calculate expected value
            expected_return = prob_win * cash_value
            net_ev = expected_return - ticket_cost
            
            # Store the results back in the competition data
            comp['Net EV'] = round(net_ev, 4)
            comp['Expected Return'] = round(expected_return, 4)
            comp['Probability'] = f"1 in {int(remaining_tickets):,}"
            comp['Expected Remaining Tickets'] = int(remaining_tickets)
            comp['Cash Alternative'] = cash_value  # Standardize the field name
            
            # Include ALL competitions (positive and negative EV)
            all_ev_comps.append(comp)
            hours_remaining = comp.get('Hours Remaining', 'Unknown')
            ev_status = "✅ Positive" if net_ev > 0 else "❌ Negative"
            print(f"{ev_status} EV: {comp['Prize'][:50]} - EV: £{net_ev:.2f} (Hours left: {hours_remaining:.1f})")
                
        except (ValueError, ZeroDivisionError) as e:
            print(f"⚠️  Error calculating EV for '{comp.get('Prize', 'Unknown')}': {e}")
            continue
    
    # Sort by EV descending
    all_ev_comps.sort(key=lambda x: x['Net EV'], reverse=True)
    
    print(f"Calculated EV for {len(all_ev_comps)} competitions")
    return all_ev_comps

def calculate_ev(competitions):
    """Calculate expected value for each competition using expected ticket sales - ONLY POSITIVE EVs"""
    print("📊 Calculating expected values...")
    
    positive_ev_comps = []
    
    for comp in competitions:
        # Check required fields
        required_fields = ['Total', 'Sold', 'Cost']
        missing_fields = [field for field in required_fields if field not in comp or comp[field] is None]
        
        if missing_fields:
            print(f"🔍 Debug - Competition '{comp.get('Prize', 'Unknown')}' has fields: {list(comp.keys())}")
            print(f"⚠️  Skipping '{comp.get('Prize', 'Unknown')}' - missing fields: {missing_fields}")
            continue
        
        # Handle both 'Cash' and 'cash_alternative' field names
        cash_value = comp.get('Cash') or comp.get('cash_alternative') or comp.get('Cash Alternative')
        
        if not cash_value or cash_value <= 0:
            print(f"⚠️  Skipping '{comp.get('Prize', 'Unknown')}' - no valid cash value: {cash_value}")
            continue
        
        try:
            # Use the correct field names
            total_tickets = float(comp['Total'])
            current_sold = float(comp['Sold']) 
            expected_sold = float(comp.get('Expected Sold', current_sold))  # Use expected if available
            ticket_cost = float(comp['Cost'])
            cash_value = float(cash_value)
            
            # Skip if any values are invalid
            if total_tickets <= 0 or ticket_cost <= 0 or cash_value <= 0:
                continue
            
            # Calculate probability of winning based on expected ticket sales
            remaining_tickets = total_tickets - expected_sold
            if remaining_tickets <= 0:
                print(f"⚠️  Skipping '{comp.get('Prize', 'Unknown')}' - expected to sell out (expected sold: {expected_sold}, total: {total_tickets})")
                continue
                
            prob_win = 1 / remaining_tickets
            
            # Calculate expected value
            expected_return = prob_win * cash_value
            net_ev = expected_return - ticket_cost
            
            # Store the results back in the competition data
            comp['Net EV'] = round(net_ev, 4)
            comp['Expected Return'] = round(expected_return, 4)
            comp['Probability'] = f"1 in {int(remaining_tickets):,}"
            comp['Expected Remaining Tickets'] = int(remaining_tickets)
            comp['Cash Alternative'] = cash_value  # Standardize the field name
            
            # Only include if EV is positive
            if net_ev > 0:
                positive_ev_comps.append(comp)
                hours_remaining = comp.get('Hours Remaining', 'Unknown')
                print(f"✅ Positive EV: {comp['Prize'][:50]} - EV: £{net_ev:.2f} (Hours left: {hours_remaining:.1f})")
                
        except (ValueError, ZeroDivisionError) as e:
            print(f"⚠️  Error calculating EV for '{comp.get('Prize', 'Unknown')}': {e}")
            continue
    
    # Sort by EV descending
    positive_ev_comps.sort(key=lambda x: x['Net EV'], reverse=True)
    
    print(f"Found {len(positive_ev_comps)} positive-EV competitions")
    return positive_ev_comps

def main():
    global driver
    
    print("Script started!")
    print(f"Current time: {datetime.now()}")
    
    try:
        now = datetime.now(TZ)
        print(f"Current time: {now}")
        hits, all_evs, user_preference_comps = fetch_and_parse(now)
        
        # Analyze cash correlation and estimate missing values
        # correlation_result = analyze_cash_correlation_and_estimate(all_evs)
        
        # Use regression approach to estimate missing cash values
        all_evs_with_estimates = estimate_cash_value_regression(all_evs)
        
        # Filter hits to only include competitions with estimated cash values
        # and add the cash estimates to the hits
        for hit in hits:
            prize = hit['Prize']
            # Find the corresponding competition in all_evs_with_estimates
            for comp_with_estimate in all_evs_with_estimates:
                if comp_with_estimate['Prize'] == prize:
                    # Only copy over estimated cash value if hit has no real cash value
                    if ('Cash' not in hit or not hit['Cash'] or hit['Cash'] == 0) and 'Cash' in comp_with_estimate:
                        hit['Cash'] = comp_with_estimate['Cash']
                        hit['cash_estimated'] = comp_with_estimate.get('cash_estimated', False)
                    break
        
        # Now proceed with EV calculations using only the 8-hour filtered competitions
        positive_ev_competitions = calculate_ev(hits)
        
        print(f"Found {len(positive_ev_competitions)} positive-EV competitions")
        
        # Debug: Show the positive EV competitions
        if positive_ev_competitions:
            print("\nPOSITIVE EV COMPETITIONS FOUND:")
            for hit in positive_ev_competitions:
                print(f"  • {hit['Prize']}")
                print(f"    EV: £{hit['Net EV']:.2f} | Cost: £{hit['Cost']:.2f}")
                if hit.get('Cash'):
                    print(f"    Cash Alternative: £{hit['Cash']:.0f}")
                print(f"    URL: {hit['URL']}")
                print()
        
        # Find the highest EV competition from all 8-hour competitions (even if negative)
        highest_ev_comp = None
        if hits:  # If we have any competitions ending within 8 hours
            all_calculated_evs = calculate_all_evs(hits)  # This includes all EVs, not just positive ones
            if all_calculated_evs:
                highest_ev_comp = all_calculated_evs[0]  # Already sorted by EV descending
                print(f"\nHighest EV competition (within 8 hours): {highest_ev_comp['Prize']} - EV: £{highest_ev_comp['Net EV']:.2f}")
        else:
            # No competitions ending within 8 hours, so look at all competitions to find the best one
            print("No competitions ending within 8 hours, checking all competitions for highest EV...")
            all_competitions_with_cash = []
            for comp in all_evs_with_estimates:
                if comp.get('Cash') and comp.get('Total') and comp.get('Sold') is not None and comp.get('Cost'):
                    all_competitions_with_cash.append(comp)
            
            if all_competitions_with_cash:
                all_calculated_evs = calculate_all_evs(all_competitions_with_cash)
                if all_calculated_evs:
                    highest_ev_comp = all_calculated_evs[0]  # Already sorted by EV descending
                    print(f"\nHighest EV competition (from all competitions): {highest_ev_comp['Prize']} - EV: £{highest_ev_comp['Net EV']:.2f}")

        # After finding positive EV competitions, add detailed analysis
        if positive_ev_competitions:
            print(f"\n📊 DETAILED ANALYSIS (Expected EV based on sales acceleration):")
            print("=" * 80)
            
            for i, comp in enumerate(positive_ev_competitions, 1):
                title = comp['Prize']
                current_sold = comp['Sold']
                expected_sold = comp.get('Expected Sold', current_sold)
                total = comp['Total']
                cost = comp['Cost']
                cash_value = comp.get('Cash Alternative', 0)
                hours_remaining = comp.get('Hours Remaining', 0)
                url = comp['URL']
                
                # Calculate based on expected sales
                expected_remaining = total - expected_sold
                probability = 1 / expected_remaining if expected_remaining > 0 else 0
                expected_value = probability * cash_value if cash_value else 0
                roi = ((expected_value - cost) / cost * 100) if cost > 0 else 0
                ev_minus_cost = expected_value - cost
                
                print(f"\n{i}. {title}")
                print("-" * 60)
                print(f"⏰ Time remaining: {hours_remaining:.1f} hours")
                print(f"🎫 Current tickets: {current_sold:,} sold / {total:,} total ({total-current_sold:,} remaining)")
                print(f"📈 Expected final: {expected_sold:,.0f} sold / {total:,} total ({expected_remaining:,.0f} remaining)")
                print(f"🚀 Sales acceleration: +{((expected_sold - current_sold) / max(1, current_sold) * 100):.1f}% additional sales expected")
                print(f"💰 Cost per ticket: £{cost:.2f}")
                print(f"🏆 Cash alternative: £{cash_value:,.0f}" if cash_value else "🏆 No cash alternative")
                print(f"📊 Win probability: {probability:.6f} ({probability*100:.4f}%)")
                print(f"💰 Expected value: £{expected_value:.2f}")
                print(f"💎 EV minus cost: £{ev_minus_cost:.2f}")
                print(f"🏁 ROI: {roi:.1f}%")
                
                # Risk assessment
                if probability > 0.01:  # >1%
                    risk = "🟢 LOW RISK"
                elif probability > 0.001:  # >0.1%
                    risk = "🟡 MEDIUM RISK"
                else:  # <0.1%
                    risk = "🔴 HIGH RISK"
                print(f"⚠️  Risk level: {risk}")
                
                # Value assessment
                if ev_minus_cost > 50:
                    value = "🌟 EXCELLENT VALUE"
                elif ev_minus_cost > 10:
                    value = "✅ GOOD VALUE"
                elif ev_minus_cost > 1:
                    value = "⚡ DECENT VALUE"
                else:
                    value = "⚠️  MARGINAL VALUE"
                print(f"💯 Value rating: {value}")
                
                # Urgency based on time remaining
                if hours_remaining < 2:
                    urgency = "🔥 URGENT - Less than 2 hours!"
                elif hours_remaining < 4:
                    urgency = "⚡ HIGH URGENCY - Less than 4 hours"
                elif hours_remaining < 6:
                    urgency = "⏰ MODERATE URGENCY - Less than 6 hours"
                else:
                    urgency = "📅 Low urgency"
                print(f"⏰ Urgency: {urgency}")
                
                print(f"🔗 URL: {url}")

            print("\n" + "=" * 80)
            print(f"📋 SUMMARY: Found {len(positive_ev_competitions)} positive EV opportunities (ending within 8 hours)")
            total_potential_profit = sum(comp.get('Net EV', 0) for comp in positive_ev_competitions)
            print(f"💰 Total potential expected profit: £{total_potential_profit:.2f}")

        df_old = load_log()
        
        # Always send email - either with positive EV comps, or with status update and highest EV comp
        if positive_ev_competitions:
            df_new = pd.DataFrame(positive_ev_competitions)
            print(f"Checking {len(df_new)} competitions against {len(df_old)} previously seen...")

            # dedupe on Prize+URL+end-window
            merged = df_new.merge(df_old, on=["Prize","URL"], how="left", indicator=True)
            fresh  = merged[merged["_merge"] == "left_only"].drop(columns=["_merge"])
            
            print(f"Found {len(fresh)} fresh competitions (not seen before)")
            
            if fresh.empty:
                print("All positive-EV competitions were already seen before")
                print("Sending email with current opportunities...")
                # Send email with current hits even if not fresh
                notify(positive_ev_competitions, user_preference_comps, highest_ev_comp=highest_ev_comp)
            else:
                # append and save
                df_combined = pd.concat([df_old, fresh[CSV_COLUMNS]], ignore_index=True)
                save_log(df_combined)
                # Send email notification
                notify(fresh.to_dict("records"), user_preference_comps, highest_ev_comp=highest_ev_comp)
                print(f"\n✅ Found {len(fresh)} NEW positive-EV competitions!")
        else:
            print("No positive-EV competitions found")
            print("Sending status email with highest EV competition...")
            # Send email with status update and highest EV comp
            notify([], user_preference_comps, highest_ev_comp=highest_ev_comp)
            
        print("📧 Email notification sent!")

    except Exception as e:
        print(f"Error in main(): {e}")

if __name__ == "__main__":
    main()
