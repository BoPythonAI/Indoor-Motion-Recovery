import json,time,os
from pathlib import Path
import numpy as np
import torch
R=Path(os.environ.get('RECOVERY_ASSET_ROOT', '/path/to/recovery-assets'))
P=Path(__file__).resolve().parents[1]
def dump(path,value):Path(path).write_text(json.dumps(value,indent=2))
from task_program import require_supported
OUT=P/'example/relation_vlm_v1'
B=R/'scene0317_015317_decision_camera_v1/proposal_grounding_v2'
TESTS=[
 {'id':'canonical','text':'Walk to the office chair farthest from the trash can. During walking and recovery, no body part may touch the table nearest your starting position.','goal_relation':'farthest','forbidden_relation':'nearest'},
 {'id':'paraphrase','text':'Head for the office chair most distant from the trash can. Keep your entire body off the table closest to where you started, both while walking and while recovering balance.','goal_relation':'farthest','forbidden_relation':'nearest'},
 {'id':'counterfactual','text':'Walk to the office chair nearest the trash can. During walking and recovery, do not touch the table farthest from your starting position with any body part.','goal_relation':'nearest','forbidden_relation':'farthest'}]

def ground(parsed,manifest,meta,arrays,instruction):
 if parsed.get('uncertainty')!='low':raise ValueError('Uncertain task parse')
 g=parsed['goal'];f=parsed['forbidden']
 if g['relation'] not in ['nearest','farthest'] or f['relation'] not in ['nearest','farthest']:raise ValueError('Unsupported relation')
 if str(f['anchor']).lower()!='start' or f['body_parts']!=['all'] or set(f['phase'])!={'walking','recovery'}:raise ValueError('Unsupported contact predicate')
 objects={o['object_id']:o for o in meta['objects']};visible={p['instance_id_audit_only'] for p in manifest['proposals'] if any(b is not None for b in p['boxes_xyxy'])}
 aliases={'chair':'office chair','bin':'trash can'}
 norm=lambda s:aliases.get(s.lower(),s.lower())
 anchor=[o for o in objects.values() if o['label'].lower()==norm(g['anchor_category'])]
 if len(anchor)!=1:raise ValueError('Ambiguous anchor category')
 center=lambda o:np.asarray(o['bounds']).mean(0)[:2]
 goals=[(float(np.linalg.norm(center(o)-center(anchor[0]))),i) for i,o in objects.items() if i in visible and o['label'].lower()==norm(g['category'])]
 start=arrays['reference_world'][0,0,:2]
 forbidden=[]
 for i,o in objects.items():
  if i not in visible or o['label'].lower()!=norm(f['category']):continue
  lo,hi=np.asarray(o['bounds'])[:,:2];forbidden.append((float(np.linalg.norm(np.maximum(np.maximum(lo-start,start-hi),0))),i))
 if not goals or not forbidden:raise ValueError('No matching visible instances')
 target=(min if g['relation']=='nearest' else max)(goals)[1];fid=(min if f['relation']=='nearest' else max)(forbidden)[1]
 return {'schema_version':'task_program_v1','scene_id':meta['scene_id'],'visual_observation_run_id':manifest['run_id'],'task_id':'relation_task','instruction':instruction,'status':'COMPILED','execution_gate':'PERMIT_DEVELOPMENT_ONLY','goal':{'instance_id_from_proposal_map':target,'predicate':'reach_and_stop'},'forbidden_instance_id':fid,'contact_events':[{'mode':'forbidden','object_instance_id_from_proposal_map':fid,'body_parts':['all'],'phase':['walking','recovery']}],'parsed_relations':parsed,'grounding_audit':{'goal_distances':goals,'forbidden_start_distances':forbidden},'source':{'model':'Qwen3-VL-4B-Instruct','role':'instruction-to-relation parsing','geometry':'known ScanNet instances and metric map','image_role':'provided as context; independent visual benefit not established'},'limits':['Restricted nearest/farthest grammar','Known object categories and map','No dynamics or object strength prediction']}

def main():
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--instruction');parser.add_argument('--scene',default=str(R/'scenes/scene0317_00_015317'));parser.add_argument('--manifest',default=str(B/'proposal_manifest_v2.json'));parser.add_argument('--out',default=str(OUT));args=parser.parse_args()
 out=Path(args.out);tests=TESTS if args.instruction is None else [{'id':'task','text':args.instruction}]
 out.mkdir(parents=True,exist_ok=True)
 if (out/'results.json').exists():raise ValueError('Existing evidence')
 dump(out/'tests_predeclared.json',tests)
 m=json.loads(Path(args.manifest).read_text())[0];scene=Path(args.scene);meta=json.loads((scene/'scene_metadata.json').read_text());arrays=np.load(scene/'scene_arrays.npz')
 from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
 torch.set_num_threads(2);mp=Path(os.environ['QWEN_MODEL_PATH'])
 model=Qwen3VLForConditionalGeneration.from_pretrained(str(mp),dtype=torch.bfloat16,local_files_only=True).to('cuda').eval();proc=AutoProcessor.from_pretrained(str(mp),local_files_only=True)
 rows=[]
 for test in tests:
  schema={'goal':{'category':'scene category','relation':'nearest or farthest','anchor_category':'scene category'},'forbidden':{'category':'scene category','relation':'nearest or farthest','anchor':'start','body_parts':['all'],'phase':['walking','recovery']},'uncertainty':'low or high'}
  prompt=('Convert the instruction into symbolic task relations. Available relevant category names: office chair, trash can, table. Do NOT choose object IDs or estimate distances: a metric map will resolve the relations. The schema shows allowed value types, not answers. Do not infer contact permission from geometry or color. Return only JSON matching '+json.dumps(schema)+'. Instruction: '+test['text'])
  content=[{'type':'image','image':s} for s in m['annotated_images'][:2]]+[{'type':'text','text':prompt}]
  x=proc.apply_chat_template([{'role':'user','content':content}],tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt').to(model.device);t=time.perf_counter()
  with torch.inference_mode():y=model.generate(**x,max_new_tokens=220,do_sample=False)
  raw=proc.batch_decode([o[len(i):] for i,o in zip(x.input_ids,y)],skip_special_tokens=True)[0];row={'id':test['id'],'prompt':prompt,'raw':raw,'latency_s':time.perf_counter()-t}
  try:
   parsed=json.loads(raw.strip().removeprefix('```json').removesuffix('```').strip());program=ground(parsed,m,meta,arrays,test['text'])
   row.update(parsed=parsed,program=program,operator_correct=(parsed['goal']['relation']==test['goal_relation'] and parsed['forbidden']['relation']==test['forbidden_relation']) if 'goal_relation' in test else None)
   try:require_supported(program,meta);row['motion_reference_compatible']=True
   except ValueError:row['motion_reference_compatible']=False
   dump(out/(test['id']+'_program.json'),program)
  except Exception as e:row.update(status='ABSTAIN',error=type(e).__name__+': '+str(e))
  rows.append(row);print(json.dumps({'id':row['id'],'operator_correct':row.get('operator_correct'),'target':row.get('program',{}).get('goal',{}).get('instance_id_from_proposal_map'),'forbidden':row.get('program',{}).get('forbidden_instance_id'),'compatible':row.get('motion_reference_compatible')}),flush=True)
 dump(out/'results.json',rows)
if __name__=='__main__':main()
