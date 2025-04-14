import math

# 计算两点之间的距离
def distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# 计算三角形ABC的外接圆的圆心和半径
def circumcenter(xa, ya, xb, yb, xc, yc):
    # 计算边的中点
    D = 2 * (xa * (yb - yc) + xb * (yc - ya) + xc * (ya - yb))
    ux = ((xa**2 + ya**2) * (yb - yc) + (xb**2 + yb**2) * (yc - ya) + (xc**2 + yc**2) * (ya - yb)) / D
    uy = ((xa**2 + ya**2) * (xc - xb) + (xb**2 + yb**2) * (xa - xc) + (xc**2 + yc**2) * (xb - xa)) / D
    # 圆心 (ux, uy) 半径
    r = distance(ux, uy, xa, ya)
    return ux, uy, r

# 计算两个圆的交集面积
def intersection_area(x1, y1, r1, x2, y2, r2):
    d = distance(x1, y1, x2, y2)
    
    if d >= r1 + r2:  # 没有交集
        return 0
    if d <= abs(r1 - r2):  # 一个圆完全包含另一个
        return math.pi * min(r1, r2) ** 2
    
    # 计算交集面积
    part1 = r1**2 * math.acos((d**2 + r1**2 - r2**2) / (2 * d * r1))
    part2 = r2**2 * math.acos((d**2 + r2**2 - r1**2) / (2 * d * r2))
    part3 = 0.5 * math.sqrt((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2))
    
    return part1 + part2 - part3

# 主函数
def main():
    # 读取输入
    xa, ya, xb, yb, xc, yc = map(int, input().split())
    xd, yd, xe, ye, xf, yf = map(int, input().split())
    
    # 计算两个三角形的外接圆
    O_x, O_y, O_r = circumcenter(xa, ya, xb, yb, xc, yc)
    P_x, P_y, P_r = circumcenter(xd, yd, xe, ye, xf, yf)
    
    # 计算两个圆的交集面积
    result = intersection_area(O_x, O_y, O_r, P_x, P_y, P_r)
    
    # 输出结果，保留六位小数
    print(f"{result:.6f}")

if __name__ == "__main__":
    main()
