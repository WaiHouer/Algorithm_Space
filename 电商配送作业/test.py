import random

print(f'佩戴口罩1-{random.randint(20,31)}')
x = random.randint(23,31)
print(f'隔离1-{x}')
if random.randint(0,10) >= 5:
    print(f'社交距离5-{random.randint(1,30)}')
else:
    print(f'社交距离4-{random.randint(20,31)}')
if random.randint(0,10) >= 3:
    print(f'核酸检测2-{random.randint(1,15)}')
else:
    print(f'核酸检测1-{random.randint(25,31)}')
print(f'疫苗接种12-{random.randint(10,31)}')
print(f'把控入境关口1-{x}')
