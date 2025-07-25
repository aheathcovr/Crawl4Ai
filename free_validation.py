"""
Free Validation System for Healthcare Facility Data
Uses only free/open source tools and APIs for data verification
Zero cost validation with comprehensive coverage
"""

import asyncio
import logging
import re
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import aiohttp
import phonenumbers
from phonenumbers import geocoder, carrier, PhoneNumberType
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import validators
from urllib.parse import urlparse
import time


@dataclass
class ValidationResult:
    """Result of validation for a single field"""
    field_name: str
    original_value: str
    is_valid: bool
    confidence: float  # 0.0 to 1.0
    validation_method: str
    corrected_value: Optional[str] = None
    error_message: Optional[str] = None
    additional_info: Dict[str, Any] = None


@dataclass
class FacilityValidationReport:
    """Complete validation report for a facility"""
    facility_name: str
    source_url: str
    validation_date: str
    overall_confidence: float
    field_validations: List[ValidationResult]
    flags: List[str]  # Warning flags
    is_likely_valid: bool
    validation_summary: Dict[str, Any]


class PhoneValidator:
    """Free phone number validation using phonenumbers library"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate(self, phone: str, country_code: str = "US") -> ValidationResult:
        """Validate phone number format and extract information"""
        
        if not phone:
            return ValidationResult(
                field_name="phone",
                original_value="",
                is_valid=False,
                confidence=0.0,
                validation_method="format_check",
                error_message="Empty phone number"
            )
        
        try:
            # Clean the phone number
            cleaned_phone = re.sub(r'[^\d+()]', '', phone)
            
            # Parse the phone number
            parsed_number = phonenumbers.parse(cleaned_phone, country_code)
            
            # Validate the number
            is_valid = phonenumbers.is_valid_number(parsed_number)
            is_possible = phonenumbers.is_possible_number(parsed_number)
            
            # Get additional information
            location = geocoder.description_for_number(parsed_number, "en")
            carrier_name = carrier.name_for_number(parsed_number, "en")
            number_type = phonenumbers.number_type(parsed_number)
            
            # Format the number properly
            formatted_number = phonenumbers.format_number(
                parsed_number, 
                phonenumbers.PhoneNumberFormat.NATIONAL
            )
            
            # Determine confidence
            confidence = 0.0
            if is_valid:
                confidence = 0.95
            elif is_possible:
                confidence = 0.7
            else:
                confidence = 0.3
            
            # Check if it's a mobile number (lower confidence for healthcare facilities)
            if number_type == PhoneNumberType.MOBILE:
                confidence *= 0.8  # Reduce confidence for mobile numbers
            
            return ValidationResult(
                field_name="phone",
                original_value=phone,
                is_valid=is_valid,
                confidence=confidence,
                validation_method="phonenumbers_library",
                corrected_value=formatted_number if is_valid else None,
                additional_info={
                    "is_possible": is_possible,
                    "location": location,
                    "carrier": carrier_name,
                    "number_type": self._get_number_type_name(number_type),
                    "country_code": parsed_number.country_code,
                    "national_number": parsed_number.national_number
                }
            )
            
        except phonenumbers.NumberParseException as e:
            return ValidationResult(
                field_name="phone",
                original_value=phone,
                is_valid=False,
                confidence=0.1,
                validation_method="phonenumbers_library",
                error_message=f"Parse error: {e}"
            )
        except Exception as e:
            self.logger.error(f"Phone validation error: {e}")
            return ValidationResult(
                field_name="phone",
                original_value=phone,
                is_valid=False,
                confidence=0.0,
                validation_method="phonenumbers_library",
                error_message=f"Validation error: {e}"
            )
    
    def _get_number_type_name(self, number_type) -> str:
        """Convert number type enum to readable string"""
        type_names = {
            PhoneNumberType.FIXED_LINE: "Fixed Line",
            PhoneNumberType.MOBILE: "Mobile",
            PhoneNumberType.FIXED_LINE_OR_MOBILE: "Fixed Line or Mobile",
            PhoneNumberType.TOLL_FREE: "Toll Free",
            PhoneNumberType.PREMIUM_RATE: "Premium Rate",
            PhoneNumberType.SHARED_COST: "Shared Cost",
            PhoneNumberType.VOIP: "VoIP",
            PhoneNumberType.PERSONAL_NUMBER: "Personal Number",
            PhoneNumberType.PAGER: "Pager",
            PhoneNumberType.UAN: "UAN",
            PhoneNumberType.VOICEMAIL: "Voicemail",
            PhoneNumberType.UNKNOWN: "Unknown"
        }
        return type_names.get(number_type, "Unknown")


class AddressValidator:
    """Free address validation using geopy and regex patterns"""
    
    def __init__(self):
        self.geolocator = Nominatim(user_agent="healthcare_facility_validator")
        self.logger = logging.getLogger(__name__)
        self.rate_limit_delay = 1.0  # Respect Nominatim rate limits
        self.last_request_time = 0
    
    async def validate(self, address: str, city: str = None, state: str = None, zip_code: str = None) -> ValidationResult:
        """Validate address using format checking and geocoding"""
        
        if not address:
            return ValidationResult(
                field_name="address",
                original_value="",
                is_valid=False,
                confidence=0.0,
                validation_method="format_check",
                error_message="Empty address"
            )
        
        # Combine address components
        full_address = self._build_full_address(address, city, state, zip_code)
        
        # Format validation first
        format_result = self._validate_address_format(full_address)
        
        # Geocoding validation (with rate limiting)
        geocoding_result = await self._validate_with_geocoding(full_address)
        
        # Combine results
        is_valid = format_result["is_valid"] and geocoding_result["is_valid"]
        confidence = (format_result["confidence"] + geocoding_result["confidence"]) / 2
        
        return ValidationResult(
            field_name="address",
            original_value=address,
            is_valid=is_valid,
            confidence=confidence,
            validation_method="format_and_geocoding",
            corrected_value=geocoding_result.get("formatted_address"),
            additional_info={
                "format_validation": format_result,
                "geocoding_validation": geocoding_result,
                "full_address_used": full_address
            }
        )
    
    def _build_full_address(self, address: str, city: str = None, state: str = None, zip_code: str = None) -> str:
        """Build complete address string from components"""
        parts = [address]
        if city:
            parts.append(city)
        if state:
            parts.append(state)
        if zip_code:
            parts.append(zip_code)
        return ", ".join(parts)
    
    def _validate_address_format(self, address: str) -> Dict[str, Any]:
        """Validate address format using regex patterns"""
        
        # US address patterns
        patterns = {
            "street_number": r'^\d+',
            "street_name": r'\d+\s+([A-Za-z0-9\s]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Boulevard|Blvd|Lane|Ln|Way|Circle|Cir|Court|Ct|Place|Pl))',
            "city": r'([A-Za-z\s]+),\s*[A-Z]{2}',
            "state": r'\b[A-Z]{2}\b',
            "zip_code": r'\b\d{5}(?:-\d{4})?\b'
        }
        
        validations = {}
        for component, pattern in patterns.items():
            match = re.search(pattern, address, re.IGNORECASE)
            validations[component] = {
                "found": match is not None,
                "value": match.group(1) if match and match.groups() else match.group(0) if match else None
            }
        
        # Calculate confidence based on components found
        required_components = ["street_number", "street_name"]
        optional_components = ["city", "state", "zip_code"]
        
        required_score = sum(1 for comp in required_components if validations[comp]["found"])
        optional_score = sum(1 for comp in optional_components if validations[comp]["found"])
        
        confidence = (required_score / len(required_components)) * 0.7 + (optional_score / len(optional_components)) * 0.3
        is_valid = required_score == len(required_components)
        
        return {
            "is_valid": is_valid,
            "confidence": confidence,
            "components": validations,
            "method": "regex_pattern_matching"
        }
    
    async def _validate_with_geocoding(self, address: str) -> Dict[str, Any]:
        """Validate address using free geocoding service"""
        
        try:
            # Rate limiting
            current_time = time.time()
            if current_time - self.last_request_time < self.rate_limit_delay:
                await asyncio.sleep(self.rate_limit_delay - (current_time - self.last_request_time))
            
            self.last_request_time = time.time()
            
            # Geocode the address
            location = self.geolocator.geocode(address, timeout=10)
            
            if location:
                return {
                    "is_valid": True,
                    "confidence": 0.9,
                    "formatted_address": location.address,
                    "latitude": location.latitude,
                    "longitude": location.longitude,
                    "method": "nominatim_geocoding"
                }
            else:
                return {
                    "is_valid": False,
                    "confidence": 0.2,
                    "error": "Address not found in geocoding service",
                    "method": "nominatim_geocoding"
                }
                
        except (GeocoderTimedOut, GeocoderServiceError) as e:
            self.logger.warning(f"Geocoding service error: {e}")
            return {
                "is_valid": False,
                "confidence": 0.5,  # Neutral confidence when service fails
                "error": f"Geocoding service error: {e}",
                "method": "nominatim_geocoding"
            }
        except Exception as e:
            self.logger.error(f"Geocoding error: {e}")
            return {
                "is_valid": False,
                "confidence": 0.0,
                "error": f"Geocoding error: {e}",
                "method": "nominatim_geocoding"
            }


class WebsiteValidator:
    """Free website validation using HTTP requests and format checking"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.session = None
    
    async def validate(self, url: str) -> ValidationResult:
        """Validate website URL format and accessibility"""
        
        if not url:
            return ValidationResult(
                field_name="website",
                original_value="",
                is_valid=False,
                confidence=0.0,
                validation_method="format_check",
                error_message="Empty URL"
            )
        
        # Format validation
        format_result = self._validate_url_format(url)
        
        # Accessibility check
        accessibility_result = await self._check_url_accessibility(url)
        
        # Combine results
        is_valid = format_result["is_valid"] and accessibility_result["is_valid"]
        confidence = (format_result["confidence"] + accessibility_result["confidence"]) / 2
        
        return ValidationResult(
            field_name="website",
            original_value=url,
            is_valid=is_valid,
            confidence=confidence,
            validation_method="format_and_accessibility",
            corrected_value=accessibility_result.get("final_url"),
            additional_info={
                "format_validation": format_result,
                "accessibility_validation": accessibility_result
            }
        )
    
    def _validate_url_format(self, url: str) -> Dict[str, Any]:
        """Validate URL format"""
        
        try:
            # Use validators library for comprehensive URL validation
            is_valid = validators.url(url)
            
            if is_valid:
                parsed = urlparse(url)
                return {
                    "is_valid": True,
                    "confidence": 0.95,
                    "scheme": parsed.scheme,
                    "domain": parsed.netloc,
                    "path": parsed.path,
                    "method": "validators_library"
                }
            else:
                return {
                    "is_valid": False,
                    "confidence": 0.1,
                    "error": "Invalid URL format",
                    "method": "validators_library"
                }
                
        except Exception as e:
            return {
                "is_valid": False,
                "confidence": 0.0,
                "error": f"URL validation error: {e}",
                "method": "validators_library"
            }
    
    async def _check_url_accessibility(self, url: str, timeout: int = 10) -> Dict[str, Any]:
        """Check if URL is accessible"""
        
        try:
            if not self.session:
                timeout_config = aiohttp.ClientTimeout(total=timeout)
                self.session = aiohttp.ClientSession(timeout=timeout_config)
            
            async with self.session.get(url, allow_redirects=True) as response:
                
                # Extract page title if possible
                title = None
                if response.content_type and 'text/html' in response.content_type:
                    try:
                        content = await response.text()
                        title_match = re.search(r'<title>(.*?)</title>', content, re.IGNORECASE)
                        if title_match:
                            title = title_match.group(1).strip()
                    except:
                        pass
                
                return {
                    "is_valid": response.status == 200,
                    "confidence": 0.9 if response.status == 200 else 0.3,
                    "status_code": response.status,
                    "final_url": str(response.url),
                    "content_type": response.content_type,
                    "title": title,
                    "response_time": None,  # Could add timing if needed
                    "method": "http_request"
                }
                
        except asyncio.TimeoutError:
            return {
                "is_valid": False,
                "confidence": 0.2,
                "error": "Request timeout",
                "method": "http_request"
            }
        except Exception as e:
            return {
                "is_valid": False,
                "confidence": 0.1,
                "error": f"Accessibility check failed: {e}",
                "method": "http_request"
            }
    
    async def close(self):
        """Close the HTTP session"""
        if self.session:
            await self.session.close()


class EmailValidator:
    """Free email validation using regex and format checking"""
    
    def validate(self, email: str) -> ValidationResult:
        """Validate email format"""
        
        if not email:
            return ValidationResult(
                field_name="email",
                original_value="",
                is_valid=False,
                confidence=0.0,
                validation_method="format_check",
                error_message="Empty email"
            )
        
        try:
            # Use validators library
            is_valid = validators.email(email)
            
            if is_valid:
                # Additional checks for healthcare facility emails
                domain = email.split('@')[1].lower()
                
                # Check for common healthcare domains
                healthcare_domains = [
                    'gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com'  # Generic domains (lower confidence)
                ]
                
                confidence = 0.9
                if domain in healthcare_domains:
                    confidence = 0.7  # Lower confidence for generic email providers
                
                return ValidationResult(
                    field_name="email",
                    original_value=email,
                    is_valid=True,
                    confidence=confidence,
                    validation_method="validators_library",
                    additional_info={
                        "domain": domain,
                        "is_generic_provider": domain in healthcare_domains
                    }
                )
            else:
                return ValidationResult(
                    field_name="email",
                    original_value=email,
                    is_valid=False,
                    confidence=0.1,
                    validation_method="validators_library",
                    error_message="Invalid email format"
                )
                
        except Exception as e:
            return ValidationResult(
                field_name="email",
                original_value=email,
                is_valid=False,
                confidence=0.0,
                validation_method="validators_library",
                error_message=f"Email validation error: {e}"
            )


class FreeValidationSystem:
    """Complete free validation system for healthcare facility data"""
    
    def __init__(self):
        self.phone_validator = PhoneValidator()
        self.address_validator = AddressValidator()
        self.website_validator = WebsiteValidator()
        self.email_validator = EmailValidator()
        self.logger = logging.getLogger(__name__)
        
        # Validation caches to avoid redundant checks
        self.validation_cache: Dict[str, ValidationResult] = {}
        self.seen_signatures: Set[str] = set()
        
        # Stats
        self.validation_stats = {
            "total_validated": 0,
            "cache_hits": 0,
            "geocoding_calls": 0,
            "skipped_duplicates": 0
        }
    
    async def validate_facility(self, facility_data: Dict[str, Any]) -> FacilityValidationReport:
        """Validate all fields of a healthcare facility"""
        
        # Check if we've already validated this exact facility (short-circuit)
        signature = self._create_facility_signature(facility_data)
        if signature in self.seen_signatures:
            self.validation_stats["skipped_duplicates"] += 1
            self.logger.debug(f"Skipping duplicate validation for {facility_data.get('name', 'Unknown')}")
            return self._create_cached_report(facility_data)
        
        self.seen_signatures.add(signature)
        self.validation_stats["total_validated"] += 1
        
        validation_results = []
        flags = []
        
        # Validate phone number
        if facility_data.get("phone"):
            phone_result = self.phone_validator.validate(facility_data["phone"])
            validation_results.append(phone_result)
            
            if not phone_result.is_valid:
                flags.append("invalid_phone_format")
            elif phone_result.additional_info and phone_result.additional_info.get("number_type") == "Mobile":
                flags.append("mobile_phone_number")
        
        # Validate address
        if facility_data.get("address"):
            address_result = await self.address_validator.validate(
                facility_data.get("address", ""),
                facility_data.get("city", ""),
                facility_data.get("state", ""),
                facility_data.get("zip_code", "")
            )
            validation_results.append(address_result)
            
            if not address_result.is_valid:
                flags.append("invalid_address")
        
        # Validate website
        if facility_data.get("website"):
            website_result = await self.website_validator.validate(facility_data["website"])
            validation_results.append(website_result)
            
            if not website_result.is_valid:
                flags.append("invalid_website")
        
        # Validate email
        if facility_data.get("email"):
            email_result = self.email_validator.validate(facility_data["email"])
            validation_results.append(email_result)
            
            if not email_result.is_valid:
                flags.append("invalid_email")
            elif email_result.additional_info and email_result.additional_info.get("is_generic_provider"):
                flags.append("generic_email_provider")
        
        # Calculate overall confidence
        confidences = [result.confidence for result in validation_results if result.confidence > 0]
        overall_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        # Determine if facility is likely valid
        is_likely_valid = (
            overall_confidence > 0.6 and
            len([r for r in validation_results if r.is_valid]) >= len(validation_results) * 0.5
        )
        
        # Create validation summary
        validation_summary = {
            "total_fields_validated": len(validation_results),
            "valid_fields": len([r for r in validation_results if r.is_valid]),
            "invalid_fields": len([r for r in validation_results if not r.is_valid]),
            "average_confidence": overall_confidence,
            "flags_count": len(flags)
        }
        
        return FacilityValidationReport(
            facility_name=facility_data.get("name", "Unknown"),
            source_url=facility_data.get("source_url", ""),
            validation_date=datetime.now().isoformat(),
            overall_confidence=overall_confidence,
            field_validations=validation_results,
            flags=flags,
            is_likely_valid=is_likely_valid,
            validation_summary=validation_summary
        )
    
    async def validate_batch(self, facilities: List[Dict[str, Any]]) -> List[FacilityValidationReport]:
        """Validate a batch of facilities"""
        
        reports = []
        
        for i, facility in enumerate(facilities):
            self.logger.info(f"Validating facility {i+1}/{len(facilities)}: {facility.get('name', 'Unknown')}")
            
            try:
                report = await self.validate_facility(facility)
                reports.append(report)
                
                # Add delay to respect rate limits
                if i < len(facilities) - 1:  # Don't delay after the last item
                    await asyncio.sleep(1.0)
                    
            except Exception as e:
                self.logger.error(f"Validation failed for facility {facility.get('name', 'Unknown')}: {e}")
                # Create error report
                error_report = FacilityValidationReport(
                    facility_name=facility.get("name", "Unknown"),
                    source_url=facility.get("source_url", ""),
                    validation_date=datetime.now().isoformat(),
                    overall_confidence=0.0,
                    field_validations=[],
                    flags=["validation_error"],
                    is_likely_valid=False,
                    validation_summary={"error": str(e)}
                )
                reports.append(error_report)
        
        return reports
    
    async def close(self):
        """Close any open connections"""
        await self.website_validator.close()
    
    def generate_validation_summary(self, reports: List[FacilityValidationReport]) -> Dict[str, Any]:
        """Generate summary statistics for validation results"""
        
        if not reports:
            return {"error": "No validation reports provided"}
        
        total_facilities = len(reports)
        valid_facilities = len([r for r in reports if r.is_likely_valid])
        
        # Confidence distribution
        confidences = [r.overall_confidence for r in reports]
        avg_confidence = sum(confidences) / len(confidences)
        
        # Flag analysis
        all_flags = []
        for report in reports:
            all_flags.extend(report.flags)
        
        flag_counts = {}
        for flag in all_flags:
            flag_counts[flag] = flag_counts.get(flag, 0) + 1
        
        # Field validation statistics
        field_stats = {}
        for report in reports:
            for validation in report.field_validations:
                field_name = validation.field_name
                if field_name not in field_stats:
                    field_stats[field_name] = {"total": 0, "valid": 0, "avg_confidence": 0}
                
                field_stats[field_name]["total"] += 1
                if validation.is_valid:
                    field_stats[field_name]["valid"] += 1
                field_stats[field_name]["avg_confidence"] += validation.confidence
        
        # Calculate averages
        for field_name, stats in field_stats.items():
            if stats["total"] > 0:
                stats["avg_confidence"] /= stats["total"]
                stats["validation_rate"] = stats["valid"] / stats["total"]
        
        return {
            "total_facilities": total_facilities,
            "valid_facilities": valid_facilities,
            "validation_rate": valid_facilities / total_facilities,
            "average_confidence": avg_confidence,
            "flag_distribution": flag_counts,
            "field_validation_stats": field_stats,
            "confidence_distribution": {
                "high_confidence": len([c for c in confidences if c > 0.8]),
                "medium_confidence": len([c for c in confidences if 0.5 < c <= 0.8]),
                "low_confidence": len([c for c in confidences if c <= 0.5])
            }
        }


# CLI interface for validation
async def main():
    """CLI interface for free validation system"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Free Healthcare Facility Data Validation')
    parser.add_argument('--input', required=True, help='Input JSON file with facility data')
    parser.add_argument('--output', help='Output file for validation report')
    parser.add_argument('--summary', action='store_true', help='Generate summary statistics')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Load facility data
    with open(args.input, 'r') as f:
        facilities = json.load(f)
    
    if not isinstance(facilities, list):
        facilities = [facilities]
    
    # Initialize validation system
    validator = FreeValidationSystem()
    
    try:
        # Validate facilities
        print(f"Validating {len(facilities)} facilities...")
        reports = await validator.validate_batch(facilities)
        
        # Generate summary
        if args.summary:
            summary = validator.generate_validation_summary(reports)
            print(f"\nValidation Summary:")
            print(f"Total facilities: {summary['total_facilities']}")
            print(f"Valid facilities: {summary['valid_facilities']} ({summary['validation_rate']:.1%})")
            print(f"Average confidence: {summary['average_confidence']:.2f}")
            print(f"Common flags: {summary['flag_distribution']}")
        
        # Save results
        if args.output:
            output_data = {
                "validation_reports": [asdict(report) for report in reports],
                "summary": validator.generate_validation_summary(reports) if args.summary else None
            }
            
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            print(f"Validation results saved to {args.output}")
    
    finally:
        await validator.close()


if __name__ == '__main__':
    asyncio.run(main())

