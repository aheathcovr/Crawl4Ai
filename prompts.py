"""
Centralized prompt templates for LLM-based extraction
Includes positive and negative examples for better accuracy
"""

# Site Structure Analysis Prompt
SITE_STRUCTURE_ANALYSIS = """
Analyze this healthcare website to understand its structure and identify where facility information is located.

Website URL: {url}
Page Content: {page_content}

TASK: Identify the site type and navigation strategy for finding healthcare facilities.

POSITIVE INDICATORS for facility pages:
- URLs containing: /locations, /facilities, /communities, /directory, /find-a-location
- Navigation links labeled: "Our Locations", "Find a Facility", "Communities", "Care Centers"
- Content patterns: facility lists, location cards, address blocks, "View Details" links
- Interactive elements: location search, zip code finder, map views

NEGATIVE INDICATORS (avoid these):
- Corporate pages: /about-us, /leadership, /investors, /careers
- Service pages: /services, /programs, /insurance
- Contact forms without specific facility info
- Blog posts, news articles, press releases
- Generic "Contact Us" pages (corporate office only)

Please analyze and provide a JSON response with:
{
    "site_type": "corporate_chain" | "individual_facility" | "directory",
    "navigation_targets": [
        {
            "url": "full URL to target page",
            "page_type": "facility_listing" | "individual_facility" | "search_page",
            "confidence": 0.0-1.0,
            "description": "what's expected on this page",
            "css_selectors": ["suggested selectors for facility containers"]
        }
    ],
    "expected_facility_count": estimated number,
    "navigation_strategy": "description of best approach",
    "confidence": 0.0-1.0
}

Focus on pages that list multiple facilities or contain detailed facility information.
Return only valid JSON.
"""

# Facility Extraction Prompt
FACILITY_EXTRACTION = """
You are an expert at extracting healthcare facility information from web pages.

TASK: Extract detailed information about healthcare facilities from the provided content.

FACILITY TYPES TO EXTRACT:
- Skilled Nursing Facilities
- Assisted Living Communities  
- Memory Care Centers
- Continuing Care Retirement Communities (CCRC)
- Rehabilitation Centers
- Long-term Care Facilities
- Senior Living Communities

REQUIRED INFORMATION (extract when available):
- Facility name (REQUIRED)
- Complete street address
- City, State, ZIP
- Phone numbers (main, admissions, fax)
- Administrator/Director names
- Bed count or capacity
- License numbers
- Services offered
- Medicare/Medicaid certification

EXTRACTION RULES:
1. Extract ONLY actual healthcare facilities, not:
   - Corporate offices
   - Administrative buildings
   - Physician offices (unless part of a facility)
   - Outpatient clinics (unless part of a facility)

2. Each facility must have AT MINIMUM:
   - A specific name (not just "Our Location")
   - A physical address OR phone number

3. DO NOT:
   - Infer or generate missing information
   - Duplicate facilities
   - Include placeholder or example data
   - Extract job listings or staff directories

4. For addresses:
   - Include suite/unit numbers
   - Preserve original formatting
   - Don't assume state if not provided

CONFIDENCE SCORING:
- High (0.9-1.0): Name + full address + phone
- Medium (0.7-0.89): Name + partial address OR phone  
- Low (0.5-0.69): Name + minimal location info
- Skip if confidence < 0.5

{additional_instructions}

Return a JSON object with an array of facilities matching the provided schema.
If no valid facilities are found, return {{"facilities": []}}.
"""

# Facility Listing Discovery Prompt
FACILITY_LISTING_DISCOVERY = """
Analyze this webpage to find links to individual healthcare facility pages.

TASK: Identify links that lead to detailed pages about specific healthcare facilities.

LOOK FOR:
- Facility listing pages with "View Details", "Learn More", "Visit Page" links
- Location cards or tiles with facility names
- Directory tables with facility information
- Search results with facility listings
- Maps with clickable facility markers

POSITIVE PATTERNS:
- Links with facility names in URL (/locations/sunny-acres-care)
- Consistent URL patterns across facilities
- Links within structured containers (cards, lists, tables)
- "View Details" or similar action buttons

NEGATIVE PATTERNS (EXCLUDE):
- Corporate pages (/about, /careers, /news)
- Service description pages  
- Contact forms
- Social media links
- External links (different domain)
- PDF downloads (unless facility-specific)

For each facility found, extract:
- name: The facility's name as shown
- url: Complete URL to the facility's detail page
- preview_info: Any address/phone shown in the listing

IMPORTANT:
- Return only links to individual facility pages
- Facility names should be specific (not "Location 1")
- URLs must be complete and valid
- Remove duplicates

Return URLs as complete, absolute URLs.
Format: {{"facility_links": [...]}
"""

# Schema Generation Prompt
SCHEMA_GENERATION = """
Generate an optimal extraction schema for this healthcare facility webpage.

URL: {url}
HTML Sample: {html_sample}

Create a CSS/XPath-based extraction schema specifically for healthcare facility data.

SCHEMA REQUIREMENTS:
1. Target facility information containers
2. Use robust selectors that won't break easily
3. Prefer class/id selectors over tag-only selectors
4. Include fallback selectors when possible

COMMON PATTERNS:
- Facility cards: .location-card, .facility-item, [data-facility]
- Info blocks: .contact-info, .address-block, .facility-details
- Tables: table.facilities-list, .directory-table
- Lists: ul.locations-list, .facilities-grid

FIELD MAPPING:
- name: h2, h3, .facility-name, .location-title
- address: .address, .street-address, [itemprop="address"]
- phone: .phone, tel, [href^="tel:"]
- services: .services-list, .amenities

Return only the JSON schema, no explanation.
"""

# Validation Context Prompt
VALIDATION_CONTEXT = """
Review this extracted facility data for accuracy and completeness.

Facility Data: {facility_data}
Source URL: {source_url}

VALIDATION TASKS:
1. Verify this is a real healthcare facility (not corporate office)
2. Check data completeness and format consistency
3. Flag any suspicious or placeholder data
4. Assess overall confidence (0.0-1.0)

COMMON ISSUES TO CHECK:
- Generic names ("Location 1", "Test Facility")
- Incomplete addresses (missing state/zip)
- Invalid phone formats
- Mixed facility types in one record
- Corporate address instead of facility

Return validation result as JSON:
{{
    "is_valid": true/false,
    "confidence": 0.0-1.0,
    "issues": ["list of problems found"],
    "suggestions": ["recommended fixes"]
}}
"""