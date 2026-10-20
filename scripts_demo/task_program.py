"""Visual facts -> auditable task program; no access to motion outcomes."""
import hashlib,json
from pathlib import Path

def file_hash(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def color_class(row):
 text=str(row.get('top_color','')).lower()
 if any(w in text for w in ['unknown','uncertain','unclear','cannot']):return None
 pale=any(w in text for w in ['white','pale','cream'])
 warm=any(w in text for w in ['yellow','tan','brown','warm'])
 return ('pale' if pale else 'warm') if pale!=warm else None

def compile_program(manifest_path,facts_path,direct_path,spec):
 manifest=json.loads(Path(manifest_path).read_text())[0]
 proposals={p['letter']:p for p in manifest['proposals']}
 facts_rows=json.loads(Path(facts_path).read_text())
 row=next(x for x in facts_rows if x['id']=='material_fact')
 try:facts=json.loads(row['raw'].strip().removeprefix('```json').removesuffix('```').strip())
 except (ValueError,TypeError):facts={}
 candidates=spec['candidate_proposals'];goal=spec['goal_proposal']
 if goal not in proposals or any(x not in proposals for x in candidates):raise ValueError('Unknown proposal letter')
 if spec['forbidden_attribute'] not in ['pale','warm']:raise ValueError('Prototype supports pale/warm tabletop attributes only')
 classes={p:color_class(facts.get(p,{})) for p in candidates}
 matches=[p for p,c in classes.items() if c==spec['forbidden_attribute']]
 direct_rows=json.loads(Path(direct_path).read_text())
 direct=next((r.get('parsed') for r in direct_rows if r['task']['id']==spec['direct_query_id']),None)
 letter=matches[0] if len(matches)==1 else None
 agrees=isinstance(direct,dict) and direct.get('forbidden')==letter and letter is not None
 valid=all(c is not None for c in classes.values()) and agrees and direct.get('uncertainty')=='low'
 out={'schema_version':'task_program_v1','scene_id':manifest['scene_id'],'visual_observation_run_id':manifest['run_id'],'task_id':spec['task_id'],'instruction':spec['instruction'],
  'status':'COMPILED' if valid else 'ABSTAIN','execution_gate':'PERMIT_DEVELOPMENT_ONLY' if valid else 'REQUIRE_REVIEW',
  'goal':{'proposal':goal,'instance_id_from_proposal_map':proposals[goal]['instance_id_audit_only'],'predicate':'reach_and_stop'},
  'visual_facts':facts,'attribute_classes':classes,'cross_query':{'direct':direct,'fact_compiler_letter':letter,'agrees':agrees},
  'source':{'model':'Qwen3-VL-4B-Instruct','proposals':'ScanNet annotated instance proposals','goal':'explicit user proposal ID',
   'manifest_sha256':file_hash(manifest_path),'facts_sha256':file_hash(facts_path),'direct_sha256':file_hash(direct_path)},
  'limits':['Typed instruction template; not unrestricted language understanding','Appearance is visual; contact permission comes from task','No material strength, friction or support feasibility inferred','Only one all-body forbidden instance is enforced by current motion backend']}
 if valid:
  fid=proposals[letter]['instance_id_audit_only'];out['forbidden_instance_id']=fid
  out['contact_events']=[{'mode':'forbidden','object_proposal':letter,'object_instance_id_from_proposal_map':fid,'body_parts':['all'],'phase':['walking','recovery']}]
 return out

def require_supported(program,meta):
 if program.get('scene_id')!=meta['scene_id']:raise ValueError('Visual program belongs to another scene')
 if program.get('execution_gate')!='PERMIT_DEVELOPMENT_ONLY':raise ValueError('ABSTAIN: task program is unresolved')
 if program['goal']['instance_id_from_proposal_map']!=meta['target_object_id']:raise ValueError('Task goal differs from current reference task')
 ids={o['object_id'] for o in meta['objects']};fid=program['forbidden_instance_id']
 if fid not in ids:raise ValueError('Forbidden instance absent from physical scene')
 forbidden=[e for e in program['contact_events'] if e['mode']=='forbidden']
 if len(forbidden)!=1 or forbidden[0]['body_parts']!=['all']:raise ValueError('Unsupported contact rule')
 if set(forbidden[0]['phase'])!={'walking','recovery'}:raise ValueError('Unsupported temporal contact rule')
 if forbidden[0]['object_instance_id_from_proposal_map']!=fid:raise ValueError('Inconsistent forbidden IDs')
 return fid

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--manifest',required=True);p.add_argument('--facts',required=True);p.add_argument('--direct',required=True);p.add_argument('--out',required=True);a=p.parse_args()
 x=compile_program(a.manifest,a.facts,a.direct,json.loads(Path(a.spec).read_text()));Path(a.out).parent.mkdir(parents=True,exist_ok=True);Path(a.out).write_text(json.dumps(x,indent=2)+'\n');print(x['status'],x['execution_gate'])
