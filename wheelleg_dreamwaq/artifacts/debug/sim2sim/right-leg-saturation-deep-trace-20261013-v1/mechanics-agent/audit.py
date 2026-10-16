from pathlib import Path
import sys,json
import numpy as np,torch,mujoco
OUT=Path(__file__).resolve().parent
ROOT=next(p for p in OUT.parents if (p/"wheelleg_dreamwaq").is_dir())
P=ROOT/"wheelleg_dreamwaq"
sys.path.insert(0,str(P/"sim2sim/mujoco"))
sys.path.insert(0,str(P/"source/wheelleg_dreamwaq"))
import wheelleg_dreamwaq.schemas
from wheelleg_mujoco.contract import load_policy_contract,sha256_file
from wheelleg_mujoco.runner import WheelLegMujocoRuntime
from wheelleg_dreamwaq.kinematics.virtual_leg import offset_geometry_from_audit,offset_leg_fk
torch.set_num_threads(1)
E=P/"artifacts/debug/sim2sim/leg-acceleration-budget-search-20261012-v1/85/export"
XML=P/"sim2sim/mujoco/models/wheel_leg_urdf4_v1.xml"
inputs=[XML,P/"sim2sim/mujoco/model_manifest.json",P/"artifacts/phase1_v4/usd-anchor-audit.json",E/"actor.ts",E/"policy_manifest.json"]
protected={str(p):sha256_file(p) for p in inputs}
c,pm,mm=load_policy_contract(E/"policy_manifest.json",E/"actor.ts",P/"sim2sim/mujoco/model_manifest.json")
rt=WheelLegMujocoRuntime(XML,c,mm);rt.reset(np.array([0.,0.,.2]))
m,d,mp=rt.model,rt.data,rt.model_map
def bid(n):return mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,n)
def jid(n):return mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n)
def sid(n):return mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,n)
pairs=[("jIO","jAG"),("jOP","jGH"),("jwheel_left","jwheel_right"),("jIJ","jAB"),("jJM","jBE"),("jMK","jEC"),("jKN","jCF"),("jIO_dummy_child_link1","jAG_dummy_child_link1"),("jIO_dummy_child_link2","jAG_dummy_child_link2"),("jMK_dummy_child1","jEC_dummy_child_link1"),("jMK_dummy_child2","jEC_dummy_child_link2"),("jKN_dummy_child_link1","jCF_dummy_child_link1"),("jKN_dummy_child_link2","jCF_dummy_child_link2")]
S=np.diag([-1.,1.,1.]);mid=.5*(d.xanchor[jid("jIJ")]+d.xanchor[jid("jAB")])
def reflect(p):return mid+S@(p-mid)
def it(b):
    R=d.ximat[b].reshape(3,3)
    return R@np.diag(m.body_inertia[b])@R.T
bodies=[];joints=[]
for ln,rn in pairs:
    l,r=bid(ln),bid(rn);jl,jr=jid(ln),jid(rn)
    sg=1 if np.dot(d.xaxis[jr],-S@d.xaxis[jl])>=0 else -1
    bodies.append(dict(left=ln,right=rn,mass_left=float(m.body_mass[l]),mass_right=float(m.body_mass[r]),mirror_origin_error_m=float(np.linalg.norm(d.xpos[r]-reflect(d.xpos[l]))),mirror_com_error_m=float(np.linalg.norm(d.xipos[r]-reflect(d.xipos[l]))),mirror_inertia_max_error=float(np.abs(it(r)-S@it(l)@S).max())))
    joints.append(dict(left=ln,right=rn,mirror_q_sign=sg,keyframe_q_error_rad=float(d.qpos[m.jnt_qposadr[jr]]-sg*d.qpos[m.jnt_qposadr[jl]]),damping=[float(m.dof_damping[m.jnt_dofadr[j]]) for j in (jl,jr)],armature=[float(m.dof_armature[m.jnt_dofadr[j]]) for j in (jl,jr)]))
mapping=[dict(index=i,joint=n,joint_id=int(mp.joint_ids[i]),actuator_id=int(mp.actuator_ids[i]),actuator_joint_id=int(m.actuator_trnid[mp.actuator_ids[i],0]),gear=m.actuator_gear[mp.actuator_ids[i]].tolist(),ctrlrange=m.actuator_ctrlrange[mp.actuator_ids[i]].tolist(),key_q=float(d.qpos[mp.qpos_addresses[i]])) for i,n in enumerate(mp.joint_names)]
audit=json.loads((P/"artifacts/phase1_v4/usd-anchor-audit.json").read_text())
geom={s:offset_geometry_from_audit(audit["legs"][s]) for s in ("left","right")}
aa=torch.linspace(-1,1,101,dtype=torch.float64);grid=torch.cartesian_prod(aa,aa);q=torch.tensor(c.q_nominal[:2],dtype=torch.float64)+.35*grid
lf,rf=[offset_leg_fk(q,geom[s]) for s in ("left","right")];valid=lf.valid&rf.valid
delta=torch.atan2(torch.sin(lf.phi0-rf.phi0),torch.cos(lf.phi0-rf.phi0))
fg=dict(points=len(q),valid_both=int(valid.sum()),wheel_vector_max_abs_difference_m=float((lf.wheel_vector_body[valid]-rf.wheel_vector_body[valid]).abs().max()),length_max_abs_difference_m=float((lf.length[valid]-rf.length[valid]).abs().max()),phi0_max_abs_difference_deg=float(torch.rad2deg(delta[valid].abs()).max()),length_min_max_m=[float(lf.length[valid].min()),float(lf.length[valid].max())],phi0_min_max_deg=[float(torch.rad2deg(lf.phi0[valid].min())),float(torch.rad2deg(lf.phi0[valid].max()))])
samples=[]
for af,ar in [(0,0),(-1,-1),(1,1),(-1,1),(1,-1),(-.5,-.5),(.5,.5)]:
    qq=torch.tensor(c.q_nominal[:2],dtype=torch.float64)+.35*torch.tensor([af,ar],dtype=torch.float64)
    vals=[]
    for s in ("left","right"):
        f=offset_leg_fk(qq,geom[s]);vals.append(dict(side=s,length_m=float(f.length),phi0_deg=float(torch.rad2deg(f.phi0)),valid=bool(f.valid),wheel_vector_body=f.wheel_vector_body.tolist()))
    samples.append(dict(action_pair=[af,ar],target_rad=qq.tolist(),fk=vals))
jac=[]
for a in (0.,-1.,1.):
    qq=torch.tensor(c.q_nominal[:2],dtype=torch.float64)+.35*a
    def y(q):
        f=offset_leg_fk(q,geom["left"]);return torch.stack([f.length,f.phi0])
    J=torch.autograd.functional.jacobian(y,qq)
    jac.append(dict(common_action=a,jacobian_wrt_q=J.tolist(),jacobian_wrt_common_differential_radians=(J@torch.tensor([[1.,1.],[1.,-1.]],dtype=torch.float64)).tolist()))
sp=[]
for base in ("site_mk_io","site_ec_ag","site_kn_op","site_cf_gh"):
    a1,a2,b1,b2=[sid(base+x) for x in ("_a1","_a2","_b1","_b2")]
    sp.append(dict(pair=base,a_span_m=float(np.linalg.norm(d.site_xpos[a2]-d.site_xpos[a1])),b_span_m=float(np.linalg.norm(d.site_xpos[b2]-d.site_xpos[b1])),closure_error_max_m=max(float(np.linalg.norm(d.site_xpos[a1]-d.site_xpos[b1])),float(np.linalg.norm(d.site_xpos[a2]-d.site_xpos[b2])))))
signs=np.array([1.,1.,1.,1.,1.,-1.])
def response(t):
    dd=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,dd,mp.reset_keyframe_id);dd.ctrl[:]=0;dd.ctrl[mp.actuator_ids]=np.asarray(t)*signs;mujoco.mj_forward(m,dd);return dd.qacc[mp.dof_addresses]*signs,dd
q0,d0=response(np.zeros(6));responses=[]
for scale in (.5,1.,-1.):
    D=np.zeros((6,6))
    for i in range(6):
        t=np.zeros(6);t[i]=scale;qi,di=response(t);D[:,i]=(qi-q0)/scale
    perm=[2,3,0,1,5,4];Dm=D[np.ix_(perm,perm)]
    responses.append(dict(torque_scale_nm=scale,response_matrix_control_rad_s2_per_nm=D.tolist(),max_reflection_response_difference=float(np.abs(D-Dm).max()),relative_frobenius_reflection_error=float(np.linalg.norm(D-Dm)/np.linalg.norm(D))))
res=dict(schema="MechanicsReflectionAuditV1",input_hashes=protected,mapping=mapping,q_nominal=c.q_nominal.tolist(),hip_midpoint_world_m=mid.tolist(),base_com_offset_from_hip_midpoint_m=(d.xipos[mp.base_body_id]-mid).tolist(),full_com_offset_from_hip_midpoint_m=(np.sum(m.body_mass[:,None]*d.xipos,axis=0)/m.body_mass.sum()-mid).tolist(),side_mass_kg=[sum(v["mass_left"] for v in bodies),sum(v["mass_right"] for v in bodies)],body_pairs=bodies,joint_pairs=joints,fk_grid=fg,fk_samples=samples,fk_jacobians=jac,equality_spacings=sp,reset_zero_torque_qacc_control_rad_s2=q0.tolist(),reset_direct_unit_torque_response=responses,reset_ncon=int(d0.ncon),runtime_reset_metrics=rt.last_reset_metrics,production_files_unchanged=True,training_started=False,isaac_started=False)
assert all(sha256_file(Path(p))==h for p,h in protected.items())
(OUT/"results.json").write_text(json.dumps(res,indent=2,allow_nan=False),encoding="utf-8")
np.savez_compressed(OUT/"fk-grid.npz",actions=grid.numpy(),q=q.numpy(),valid=valid.numpy(),left_wheel=lf.wheel_vector_body.numpy(),right_wheel=rf.wheel_vector_body.numpy(),left_phi=lf.phi0.numpy(),right_phi=rf.phi0.numpy(),left_length=lf.length.numpy(),right_length=rf.length.numpy())
print(json.dumps({k:res[k] for k in ("side_mass_kg","fk_grid","fk_samples","fk_jacobians","equality_spacings","reset_direct_unit_torque_response","reset_ncon")},indent=2))
