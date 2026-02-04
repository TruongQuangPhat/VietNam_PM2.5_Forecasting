from geopy.geocoders import Nominatim
import time
import pandas as pd

# Initialize geolocator
geolocator = Nominatim(user_agent="my_aqi")

target_locations = [
    # --- GROUP 1: 11 PROVINCES KEPT UNCHANGED ---
    "Thành phố Hà Nội",
    "Thành phố Huế",
    "Thành phố Lai Châu",
    "Thành phố Điện Biên Phủ",
    "Thành phố Sơn La",
    "Thành phố Lạng Sơn",
    "Thành phố Hạ Long",
    "Thành phố Thanh Hóa",
    "Thành phố Vinh",
    "Thành phố Hà Tĩnh",
    "Thành phố Cao Bằng",

    # --- GROUP 2: 23 MERGED PROVINCES ---
    "Thành phố Tuyên Quang",
    "Thành phố Lào Cai",
    "Thành phố Thái Nguyên",
    "Thành phố Việt Trì",           # Phu Tho
    "Thành phố Bắc Ninh",
    "Thành phố Hưng Yên",
    "Thành phố Hải Phòng",
    "Thành phố Ninh Bình",
    "Thành phố Đông Hà",            # Quang Tri
    "Thành phố Đà Nẵng",
    "Thành phố Quảng Ngãi",
    "Thành phố Pleiku",             # Gia Lai
    "Thành phố Nha Trang",          # Khanh Hoa
    "Thành phố Đà Lạt",             # Lam Dong
    "Thành phố Buôn Ma Thuột",      # Dak Lak
    "Thành phố Hồ Chí Minh",
    "Thành phố Biên Hòa",           # Dong Nai
    "Thành phố Tây Ninh",
    "Thành phố Cần Thơ",
    "Thành phố Vĩnh Long",
    "Thành phố Cao Lãnh",           # Dong Thap
    "Thành phố Cà Mau",
    "Thành phố Long Xuyên"          # An Giang
]

print(f"Searching coordinates for {len(target_locations)} locations...")

found_data = []

def get_location_data(query):
    try:
        # Try finding in Vietnam specifically
        return geolocator.geocode(f"{query}, Vietnam")
    except:
        return None

for place in target_locations:
    location = get_location_data(place)
    
    # Retry logic: If failed, try simpler name (remove "Thành phố" or extra details)
    if not location:
        if "Thành phố" in place:
            simple_name = place.replace("Thành phố", "").strip()
            print(f"Retrying: {simple_name}...")
            location = get_location_data(simple_name)
        elif "," in place:
            simple_name = place.split(",")[0]
            print(f"Retrying: {simple_name}...")
            location = get_location_data(simple_name)

    if location:
        # Clean up name for CSV (Remove "Thành phố" for cleaner display)
        display_name = place
        if "Thành phố" in display_name:
             display_name = display_name.replace("Thành phố", "").strip()
        if "," in display_name:
             display_name = display_name.split(",")[0].strip()
        
        print(f"Found: {display_name} -> ({location.latitude}, {location.longitude})")
        found_data.append({
            "name": display_name, 
            "lat": location.latitude,
            "lon": location.longitude
        })
    else:
        print(f"Not found: {place}")
    
    time.sleep(1)

# Save result to CSV
if found_data:
    df = pd.DataFrame(found_data)
    output_file = "data/raw/vietnam_locations.csv"
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    
    print(f"Done. Data saved to '{output_file}' with {len(df)} locations.")
else:
    print("Error: No locations found.")