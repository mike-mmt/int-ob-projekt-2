from sklearn.preprocessing import StandardScaler, normalize, MinMaxScaler

a1 = 30
a2 = 211
a = a2 - a1

print(a)
angle_difference = (a + 180) % 360 - 180
print(angle_difference)

state = [430//8, 225//8, 50]

print(state)

scaler = StandardScaler()

scaler2 = MinMaxScaler()

print(scaler2.fit_transform([state]))

print(normalize([state]))

print(scaler.fit_transform([state]))

print(state)

x= 15
y=100

original_range = 485 - 15 + 1

scaling_factor = 7 / original_range

scaled_x = int(round((x - 15) * scaling_factor))
scaled_y = int(round((y - 15) * scaling_factor))

print(scaled_x, scaled_y)