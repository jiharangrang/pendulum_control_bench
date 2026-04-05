import argparse
import mujoco
import numpy as np

DEFAULT_XML = r"""
<mujoco model="inverted pendulum">
    <compiler inertiafromgeom="true"/>
    <default>
        <joint armature="0" damping="1" limited="true"/>
        <geom contype="0" friction="1 0.1 0.1" rgba="0.7 0.7 0 1"/>
        <tendon/>
        <motor ctrlrange="-3 3"/>
    </default>
    <option gravity="0 0 -9.81" integrator="RK4" timestep="0.02"/>
    <size nstack="3000"/>
    <worldbody>
        <geom name="rail" pos="0 0 0" quat="0.707 0 0.707 0" rgba="0.3 0.3 0.7 1" size="0.02 1" type="capsule"/>
        <body name="cart" pos="0 0 0">
            <joint axis="1 0 0" limited="true" name="slider" pos="0 0 0" range="-1 1" type="slide"/>
            <geom name="cart" pos="0 0 0" quat="0.707 0 0.707 0" size="0.1 0.1" type="capsule"/>
            <body name="pole" pos="0 0 0">
                <joint axis="0 1 0" name="hinge" pos="0 0 0" range="-90 90" type="hinge"/>
                <geom fromto="0 0 0 0.001 0 0.6" name="cpole" rgba="0 0.7 0.7 1" size="0.049 0.3" type="capsule"/>
            </body>
        </body>
    </worldbody>
    <actuator>
        <motor ctrllimited="true" ctrlrange="-3 3" gear="100" joint="slider" name="slide"/>
    </actuator>
</mujoco>
""".strip()


def _name(m, objtype, objid):
    name = mujoco.mj_id2name(m, objtype, objid)
    return name if name is not None else f"<id {objid}>"


def print_model_info(m):
    print("model name:", "(unavailable in MjModel; use XML root if needed)")
    print("timestep:", m.opt.timestep)
    print("integrator:", int(m.opt.integrator))
    print("gravity:", m.opt.gravity.copy())
    print("nstack:", getattr(m, "nstack", None))
    print("actuator ctrlrange:\n", m.actuator_ctrlrange.copy())
    print("actuator gear:\n", m.actuator_gear.copy())
    print("joint range:\n", m.jnt_range.copy())

    print("\nbody masses / inertia:")
    for bid in range(m.nbody):
        bname = _name(m, mujoco.mjtObj.mjOBJ_BODY, bid)
        print(f"  {bid:2d} {bname:8s} mass={m.body_mass[bid]:.6f} inertia={m.body_inertia[bid]}")

    print("\ngeom size / density / friction:")
    geom_density = getattr(m, "geom_density", None)
    for gid in range(m.ngeom):
        gname = _name(m, mujoco.mjtObj.mjOBJ_GEOM, gid)
        density_str = f"{geom_density[gid]:.6f}" if geom_density is not None else "n/a"
        print(f"  {gid:2d} {gname:8s} size={m.geom_size[gid]} density={density_str} friction={m.geom_friction[gid]}")

    print("\njoint names:")
    for jid in range(m.njnt):
        jname = _name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
        print(f"  {jid:2d} {jname:8s} type={m.jnt_type[jid]} axis={m.jnt_axis[jid]} range={m.jnt_range[jid]}")

    cart_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cart")
    pole_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pole")
    hinge_jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "hinge")

    if cart_bid != -1 and pole_bid != -1 and hinge_jid != -1:
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        cart_com = d.xipos[cart_bid].copy()
        pole_com = d.xipos[pole_bid].copy()
        hinge_pos = d.xanchor[hinge_jid].copy()
        com_dist = float(np.linalg.norm(pole_com - cart_com))

        print("\nkey parameters:")
        print("cart mass:", float(m.body_mass[cart_bid]))
        print("pole mass:", float(m.body_mass[pole_bid]))
        print("cart COM:", cart_com)
        print("pole COM:", pole_com)
        print("COM distance (cart-pole):", com_dist)
        print("hinge position:", hinge_pos)


def main():
    parser = argparse.ArgumentParser(description="Load a MuJoCo XML and print compiled model parameters.")
    parser.add_argument("--xml", type=str, default="", help="Path to XML file. If omitted, uses embedded XML.")
    args = parser.parse_args()

    if args.xml:
        m = mujoco.MjModel.from_xml_path(args.xml)
    else:
        m = mujoco.MjModel.from_xml_string(DEFAULT_XML)

    print_model_info(m)


if __name__ == "__main__":
    main()
