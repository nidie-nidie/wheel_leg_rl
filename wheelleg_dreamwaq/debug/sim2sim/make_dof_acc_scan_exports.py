import argparse
import hashlib
import json
import pathlib
import shutil

import torch
from wheelleg_mujoco.model_semantics import stable_hash


def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest().upper()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',required=True)
    args=parser.parse_args()
    run=pathlib.Path(args.run).resolve()
    scan=run/'dof_acc_weight_scan_update318'
    base_export=run/'export'
    base_actor=torch.jit.load(str(base_export/'actor.ts'),map_location='cpu').eval()
    base_manifest=json.loads((base_export/'policy_manifest.json').read_text(encoding='utf-8'))
    base_golden=torch.load(base_export/'golden_vectors.pt',map_location='cpu',weights_only=False)
    for scale in ('1','10','30','100'):
        checkpoint=scan/f'model_317_dof_acc_x{scale}.pt'
        out=scan/f'export_x{scale}'
        if out.exists():
            for p in out.iterdir():
                if p.is_dir(): shutil.rmtree(p)
                else: p.unlink()
        else:
            out.mkdir()
        actor=torch.jit.load(str(base_export/'actor.ts'),map_location='cpu').eval()
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)['model_state_dict']
        state={key: state[key if key.startswith('actor.') else 'cenet.' + key].detach().cpu() for key in actor.state_dict().keys()}
        actor.load_state_dict(state,strict=True)
        actor.save(str(out/'actor.ts'))
        history=base_golden['history'].detach().cpu()
        with torch.inference_mode(): expected=actor(history).contiguous()
        torch.save({'schema_version':base_golden['schema_version'],'seed':base_golden['seed'],
                    'history':history,'expected_action_mean':expected},out/'golden_vectors.pt')
        manifest=dict(base_manifest)
        manifest['source_checkpoint']=str(checkpoint)
        manifest['source_checkpoint_sha256']=sha256(checkpoint)
        manifest['source_completed_iterations']=318
        manifest['actor_file']='actor.ts'
        manifest['actor_sha256']=sha256(out/'actor.ts')
        manifest['golden_vectors_file']='golden_vectors.pt'
        manifest['golden_vectors_sha256']=sha256(out/'golden_vectors.pt')
        manifest.pop('manifest_hash',None)
        manifest['manifest_hash']=stable_hash(manifest)
        (out/'policy_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True),encoding='utf-8')
        print(json.dumps({'scale':scale,'actor_sha256':manifest['actor_sha256'],'manifest_hash':manifest['manifest_hash']}))

if __name__=='__main__': main()

