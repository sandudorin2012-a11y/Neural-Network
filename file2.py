import cv2
import matplotlib.pyplot as plt

img = cv2.imread("road4.jpg")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

car_data = cv2.CascadeClassifier('haarcascade_cars.xml')
stop_data = cv2.CascadeClassifier('haarcascade_stopsign.xml')

car_coords = []
stop_coords = []

try:
    car_coords = car_data.detectMultiScale(img_gray, minSize=(20, 20)).tolist()
except:
    print("No Vehicles Found")

try:
    stop_coords = stop_data.detectMultiScale(img_gray, minSize=(20, 20)).tolist()
except:
    print("No STOP Signs Found")


img_height, img_width, img_channels = img.shape
left_border = img_width / 2
right_broder = img_width

def check_foward(car_coords , stop_coords):
    if len(stop_coords) != 0:
        return False
    elif len(car_coords) ==0:
        return True
    else:
        for (x, y, width, height) in car_coords:
            if x > left_border and x + width < right_broder :
                if width / img_width > 0.15:
                    return False
                return True

result = check_foward(car_coords, stop_coords)
print(result)

for (x, y, width, height) in car_coords:
    cv2.rectangle(img_rgb, (x, y), (x + width, y + height), (255, 0, 0,), 3)

for (x, y, width, height) in stop_coords:
    cv2.rectangle(img_rgb, (x,y), (x + width, y + height), (255, 0, 0), 3)

plt.figure(figsize=(10, 6))
plt.imshow(img_rgb)
plt.title(f'Movement is allowed: {result}')
plt.show()