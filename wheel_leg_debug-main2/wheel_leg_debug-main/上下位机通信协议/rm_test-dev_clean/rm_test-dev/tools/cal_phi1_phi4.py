import math

LEG_L1 = 0.215
LEG_L2 = 0.258
LEG_L3 = LEG_L2
LEG_L4 = LEG_L1
LEG_L5 = 0.01

def calc_phi1_and_phi4(phi0: float, l0: float) -> tuple[float, float]:
    """
    通过L0和Phi0的值计算关节phi1和phi4
    
    :param phi0: phi0的值
    :param l0: l0的值
    :return: 包含 (phi1, phi4) 的元组
    :note: 用于位置控制时求逆解
    """
    
    cos_beta1 = (LEG_L1**2 + l0**2 - LEG_L2**2) / (2 * LEG_L1 * l0)
    cos_beta2 = (LEG_L4**2 + l0**2 - LEG_L3**2) / (2 * LEG_L4 * l0)

    beta1 = math.acos(cos_beta1)
    beta2 = math.acos(cos_beta2)

    phi1 = phi0 + beta1
    phi4 = phi0 - beta2

    return phi1, phi4

if __name__ == "__main__":
    phi0 = 3.1415926 / 2.0

    joint2_angle_offset = 0.19163715 + math.pi
    joint3_angle_offset = -0.19163715

    L0 = 0.2
    # L0 = 0.35
    phi1, phi4 = calc_phi1_and_phi4(phi0, L0)
    print(f"phi1: {phi1}, phi4: {phi4}")
    print(f"joint2: {phi1 - joint2_angle_offset}, joint3: {phi4 - joint3_angle_offset}")