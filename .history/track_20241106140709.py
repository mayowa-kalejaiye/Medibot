import phonenumbers
from phonenumbers import geocoder, carrier

# Parse the phone number
phone_number1 = phonenumbers.parse("+2349153818344")

# Get the geographical region
region = geocoder.description_for_number(phone_number1, "en")

# Get the carrier information
carrier_info = carrier.name_for_number(phone_number1, "en")

print("\nPhone number loaded successfully\n")
print(f"Region: {region}")
print(f"Carrier: {carrier_info}")
