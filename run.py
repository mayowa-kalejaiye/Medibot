import http.client

conn = http.client.HTTPSConnection("localhost", 5500)

headersList = {
 "Accept": "*/*",
 "User-Agent": "Thunder Client (https://www.thunderclient.com)" 
}

payload = ""

conn.request("GET", "/", payload, headersList)
response = conn.getresponse()
result = response.read()

print(result.decode("utf-8"))
