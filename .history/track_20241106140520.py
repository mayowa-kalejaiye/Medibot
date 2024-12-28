import phonenumbers
from phonenumbers import geocoder
phone_number1 = phonenumbers.parse("+2349153818344")

print("\nPhone number loaded successfully\n")
print(geocoder.description_for_number(phone_number1))
