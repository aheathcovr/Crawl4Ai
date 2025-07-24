# Healthcare Facility Scraper

A comprehensive web scraper built with crawl4ai to systematically extract information about healthcare facilities (skilled nursing, assisted living, memory care, etc.) from corporate websites. The scraper automatically discovers facility locations and gathers detailed information including addresses, facility types, contact information, and operational details.

## 🏥 Features

- **Intelligent Site Navigation**: Automatically discovers facility listing pages through sitemap analysis, navigation menu parsing, and recursive link discovery
- **Multi-Facility Type Support**: Identifies and classifies skilled nursing facilities, assisted living communities, memory care centers, and continuing care retirement communities
- **Comprehensive Data Extraction**: Extracts facility names, addresses, phone numbers, emails, facility types, services, amenities, and operational information
- **Multiple Output Formats**: Saves results in JSON, CSV, and Excel formats
- **Robust Architecture**: Handles various website structures including SPAs, pagination, and dynamic content loading
- **Respectful Crawling**: Implements delays, respects robots.txt, and includes configurable concurrency limits

## 📋 Requirements

- Python 3.11+
- crawl4ai >= 0.3.0
- playwright >= 1.40.0
- Additional dependencies listed in `requirements.txt`

## 🚀 Installation

1. **Clone or download the scraper files**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   playwright install
   ```
3. **Install system dependencies (if needed):**
   ```bash
   sudo apt-get update
   sudo apt-get install -y libgstreamer1.0-0 libgtk-4-1 libgraphene-1.0-0 libatomic1 libxslt1.1 libevent-2.1-7
   ```

## 💻 Usage

### Command Line Interface

#### Single Website Scraping
```bash
python main.py --url https://lcca.com --output ./results
```

#### Multiple Websites from File
```bash
python main.py --urls-file urls.txt --output ./results --formats json csv excel
```

#### Advanced Options
```bash
python main.py --url https://example.com \
  --output ./results \
  --formats json csv \
  --max-depth 5 \
  --max-concurrent 10 \
  --delay 2.0 \
  --log-level DEBUG \
  --log-file scraper.log
```

### URL File Format
Create a text file with one URL per line:
```
https://lcca.com
https://genesishcc.com
https://brookdaleliving.com
# This is a comment
another-healthcare-site.com  # https:// will be added automatically
```

### Python API Usage
```python
import asyncio
from healthcare_scraper import HealthcareFacilityScraper

async def scrape_facilities():
    scraper = HealthcareFacilityScraper("https://lcca.com", "./output")
    facilities = await scraper.scrape_facilities()
    
    # Save results
    scraper.save_results('json')
    scraper.save_results('csv')
    
    return facilities

# Run the scraper
facilities = asyncio.run(scrape_facilities())
print(f"Found {len(facilities)} facilities")
```

## 📊 Output Data Structure

The scraper extracts the following information for each facility:

### Basic Information
- **name**: Facility name
- **facility_type**: Type of healthcare facility (Skilled Nursing, Assisted Living, etc.)
- **description**: Facility description

### Contact Information
- **address**: Street address
- **city**: City
- **state**: State (normalized to abbreviations)
- **zip_code**: ZIP code
- **phone**: Phone number (formatted)
- **fax**: Fax number
- **email**: Email address
- **website**: Website URL

### Operational Information
- **administrator**: Administrator name
- **director**: Director name
- **medical_director**: Medical director name
- **beds**: Number of beds
- **capacity**: Facility capacity
- **license_number**: State license number
- **medicare_provider_id**: Medicare provider ID
- **medicaid_certified**: Medicaid certification status

### Services and Amenities
- **services_offered**: List of services offered
- **specialties**: List of medical specialties
- **amenities**: List of facility amenities
- **care_levels**: List of care levels provided

### Additional Information
- **hours**: Operating hours
- **visiting_hours**: Visiting hours
- **parking_info**: Parking information
- **source_url**: URL where data was extracted
- **scraped_at**: Timestamp of data extraction

## 🏗️ Architecture

The scraper consists of several key components:

### Core Components
- **`healthcare_scraper.py`**: Main scraper class with facility discovery and extraction logic
- **`navigation.py`**: Advanced site navigation and URL discovery
- **`extractors.py`**: Data extraction and parsing components
- **`config.py`**: Configuration settings and patterns
- **`main.py`**: Command-line interface

### Key Features
- **Sitemap Discovery**: Automatically finds and parses XML sitemaps
- **Navigation Analysis**: Analyzes main navigation menus and footer links
- **Recursive Crawling**: Systematically explores site architecture with depth limits
- **SPA Support**: Handles JavaScript-rendered content and dynamic loading
- **Pagination Handling**: Manages various pagination patterns
- **Data Validation**: Validates and cleans extracted data
- **Structured Data**: Extracts JSON-LD and microdata when available

## ⚙️ Configuration

Key configuration options in `config.py`:

### Crawling Settings
```python
CRAWL_CONFIG = {
    'max_depth': 4,              # Maximum crawl depth
    'max_concurrent': 8,         # Maximum concurrent requests
    'delay_between_requests': 1.0, # Delay between requests (seconds)
    'timeout': 30,               # Request timeout
    'max_pages_per_site': 500,   # Maximum pages to crawl per site
    'respect_robots_txt': True   # Respect robots.txt
}
```

### Facility Types
The scraper recognizes these facility types:
- Skilled Nursing Facilities
- Assisted Living Communities
- Memory Care Centers
- Continuing Care Retirement Communities

### Data Extraction Patterns
Configurable regex patterns for extracting:
- Phone numbers
- Addresses
- ZIP codes
- Email addresses
- License numbers
- Medicare provider IDs

## 🧪 Testing Results

### Test Environment
- **Test Site**: lcca.com (Life Care Centers of America)
- **Test Date**: July 23, 2025
- **Crawl Duration**: ~6 minutes

### Results Summary
- **✅ Successfully discovered and scraped 106 healthcare facilities**
- **📍 Geographic Coverage**: Multiple states (MO, KS, TN, NV, etc.)
- **🏥 Facility Types**: Primarily skilled nursing and rehabilitation centers
- **📊 Data Quality**: Good extraction of names, phone numbers, and facility types
- **🔗 Source URLs**: Tracked source pages for all extracted data

### Sample Extracted Data
```json
{
  "name": "Life Care Center of Burlington",
  "facility_type": "Skilled Nursing",
  "address": "601 Cross St.",
  "city": "Burlington",
  "state": "KS",
  "zip_code": "66839",
  "phone": "(620) 364-2117",
  "source_url": "https://lcca.com/locations/ks/burlington/",
  "scraped_at": "2025-07-23T19:22:24.469385"
}
```

## 🔧 Troubleshooting

### Common Issues

1. **Missing System Dependencies**
   ```bash
   # Install missing libraries
   sudo apt-get install -y libgstreamer1.0-0 libgtk-4-1 libatomic1
   ```

2. **Playwright Browser Issues**
   ```bash
   # Reinstall playwright browsers
   playwright install --force
   ```

3. **Memory Issues with Large Sites**
   - Reduce `max_concurrent` setting
   - Increase `delay_between_requests`
   - Use `max_pages_per_site` limit

4. **No Facilities Found**
   - Check if the website structure matches expected patterns
   - Increase `max_depth` for deeper crawling
   - Review logs for crawling errors

### Debugging
Enable debug logging to see detailed crawling information:
```bash
python main.py --url https://example.com --log-level DEBUG --log-file debug.log
```

## 📈 Performance Optimization

### For Large Sites
- Adjust `max_concurrent` based on server capacity
- Use appropriate `delay_between_requests` to avoid rate limiting
- Set `max_pages_per_site` to limit crawl scope
- Monitor memory usage and adjust accordingly

### For Better Data Quality
- Review and update regex patterns in `config.py`
- Enhance CSS selectors for specific site structures
- Add custom extraction logic for unique website patterns

## 🤝 Contributing

To extend the scraper for new website patterns:

1. **Add new facility type patterns** in `config.py`
2. **Update CSS selectors** for new site structures
3. **Enhance regex patterns** for better data extraction
4. **Add custom extraction logic** in `extractors.py`

## 📄 License

This project is provided as-is for educational and research purposes. Please ensure compliance with website terms of service and robots.txt when using this scraper.

## 🆘 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review log files for error details
3. Verify website accessibility and structure
4. Test with simpler sites first to isolate issues

---

**Note**: This scraper is designed to be respectful of website resources and follows best practices for web scraping. Always ensure you have permission to scrape websites and comply with their terms of service.

