import phonenumbers
from phonenumbers import geocoder, carrier

# Parse the phone number
phone_number1 = phonenumbers.parse("+23470")

# Get the geographical region
region = geocoder.description_for_number(phone_number1, "en")

# Get the carrier information
carrier_info = carrier.name_for_number(phone_number1, "en")

# Placeholder for street address lookup
# You would replace this with a call to a service that provides detailed address information
street_address = "123 Example Street, Example City"

print("\nPhone number loaded successfully\n")
print(f"Region: {region}")
print(f"Carrier: {carrier_info}")
print(f"Street Address: {street_address}")
