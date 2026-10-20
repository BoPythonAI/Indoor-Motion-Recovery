"""Executable V5 warm teacher training. Validation runs in separate processes."""
import json
import time
from pathlib import Path
import numpy as np
import torch

from warm_ppo import compute_gae, make_optimizers, ppo_update
from teacher_factory import make_warm_teacher, read_checkpoint, save_checkpoint, restore_rng


def train_or_evaluate(agent, cfg):
    from warm_task import WarmRecoveryTask
    from v5_task import evaluate
    if cfg.get('validation_motion') or cfg.get('validation_metadata'):
        raise ValueError('Use sequential run_warm_cycles.py validation; no in-place cross-scene bank swaps')
    if cfg.get('teacher_kind') != 'terrain_surface':
        raise ValueError('Surface runtime requires teacher_kind=terrain_surface, without hybrid/observer gate keys')
    for key, value in dict(success_bonus=250.,stationary_weight=1.,power_weight=.01).items():
        cfg.setdefault(key,value)
    cfg.setdefault('force_strengths',[.45,.75])  # V5 reset independently allocates 20% nominal.
    task = WarmRecoveryTask(agent,cfg)
    x = task.reset()
    state = read_checkpoint(cfg['checkpoint']) if cfg.get('checkpoint') else None
    controller = make_warm_teacher(task,cfg,state)
    if cfg['mode']=='evaluate':
        if cfg.get('method') not in ('terrain_surface','terrain_prior'):
            raise ValueError('Surface evaluate supports terrain_surface or terrain_prior')
        controller.model.eval()
        result = evaluate(task,controller,cfg)
        Path(cfg['output']).mkdir(parents=True,exist_ok=True)
        (Path(cfg['output'])/'warm_provenance.json').write_text(json.dumps(cfg,indent=2))
        return result
    if cfg['mode']=='collect':
        if state is None:
            raise ValueError('Select a validated warm checkpoint before collection')
        if cfg.get('rollin_method','noisyteacher') not in ('teacher','noisyteacher'):
            raise ValueError('This minimal warm bundle supports teacher/noisyteacher collection only')
        from collection_sampling import run_collection
        controller.model.eval()
        return run_collection(task,controller,cfg)
    if cfg['mode']!='train' or controller.prior_only:
        raise ValueError('Use mode=train with the trainable terrain_surface teacher')
    if any(row['split']!='train' for row in task.rows):
        raise ValueError('Warm PPO only trains on the fixed training source split')
    out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=True)
    if (out/'metrics.jsonl').exists() and not cfg.get('resume'):
        raise ValueError('Fresh warm training requires a new output directory')
    optimizers=make_optimizers(controller.model,cfg)
    start_update=0
    if cfg.get('resume'):
        if state is None or not state.get('optimizers'):
            raise ValueError('Resume requires a warm checkpoint with both optimizers')
        if state['config']['motion_bank_sha256']!=cfg['motion_bank_sha256']:
            raise ValueError('Training resume cannot silently switch motion banks')
        for key,opt in optimizers.items():opt.load_state_dict(state['optimizers'][key])
        start_update=state['update']
        restore_rng(state)
        x=task.reset()  # new process deliberately begins fresh episodes, never old rollout fragments.
    cfg['resume_episode_reset']=bool(cfg.get('resume'))
    (out/f'config_from_update_{start_update:04d}.json').write_text(json.dumps(cfg,indent=2))
    controller.model.train()
    started=time.perf_counter();recent=[]
    horizon=int(cfg.get('horizon',64));total_updates=int(cfg.get('updates',50))
    if total_updates<=start_update:raise ValueError('updates must exceed checkpoint update')
    if not cfg.get('resume'):save_checkpoint(out/'initial.pt',controller,optimizers,0,cfg,0.)
    final_update=start_update
    for update in range(start_update,total_updates):
        task.current_bw=.75
        histories=[];priors=[];actions=[];logs=[];values=[];rewards=[];dones=[]
        corrections=[];actual_corrections=[];clip_fractions=[];entropies=[]
        critic_only=update<int(cfg.get('critic_warmup_updates',5))
        with torch.no_grad():
            for t in range(horizon):
                dist,value=controller(x)
                prior=controller.last_prior
                action=dist.mean if critic_only else dist.sample()
                histories.append(x.detach().clone());priors.append(prior.clone());actions.append(action.clone())
                logs.append(dist.log_prob(action).sum(-1));values.append(value)
                corrections.append((dist.mean-prior).abs().mean())
                entropies.append(dist.entropy().sum(-1).mean())
                x,reward,done,info=task.step(action)
                rewards.append(reward);dones.append(done)
                actual_corrections.append(info['actual_correction'].abs().mean())
                requested=info['base_normalized']+action
                clip_fractions.append(((requested.abs()>1-1e-6)&(action!=0)).float().mean())
                if done.any():
                    ids=done.nonzero(as_tuple=True)[0]
                    recent.extend(info['success'][ids].float().cpu().tolist());recent=recent[-512:]
                    x=task.reset(ids)
            last_value=controller.model.value(x)  # Critic bootstrap never queries a new action prior.
            r,v,d=torch.stack(rewards),torch.stack(values),torch.stack(dones)
            advantage,returns=compute_gae(r,v,d,last_value,cfg.get('gamma',.99),cfg.get('gae_lambda',.95))
            advantage=advantage.flatten()
            advantage=(advantage-advantage.mean())/advantage.std(unbiased=False).clamp_min(1e-6)
            batch=dict(history=torch.stack(histories).flatten(0,1),prior_delta=torch.stack(priors).flatten(0,1),
                       action=torch.stack(actions).flatten(0,1),old_log_prob=torch.stack(logs).flatten(),
                       advantage=advantage,returns=returns.flatten())
        metrics=ppo_update(controller.model,batch,optimizers,cfg,update_actor=not critic_only)
        elapsed=time.perf_counter()-started;final_update=update+1
        metrics.update(update=final_update,seconds=elapsed,critic_only=critic_only,reward=float(r.mean()),
            recent_success=float(np.mean(recent)) if recent else None,episodes=len(recent),
            correction_mean=float(torch.stack(corrections).mean()),
            actual_correction_mean=float(torch.stack(actual_corrections).mean()),
            executed_clipped_fraction=float(torch.stack(clip_fractions).mean()),
            entropy=float(torch.stack(entropies).mean()),std=float(controller.model.logstd.exp().mean()),
            prior_abs=float(batch['prior_delta'].abs().mean()),value_mean=float(v.mean()),
            return_mean=float(returns.mean()),explained_variance=float(1-(returns-v).var()/returns.var().clamp_min(1e-6)),
            transitions_per_second=(final_update-start_update)*horizon*task.n/max(elapsed,1e-6),
            resume_episode_reset=bool(cfg.get('resume') and update==start_update))
        metrics['surface_curriculum']=task.surface_curriculum.metrics(reset=True)
        with (out/'metrics.jsonl').open('a') as stream:stream.write(json.dumps(metrics)+'\n')
        print('WARM_TRAIN',json.dumps(metrics),flush=True)
        if final_update%int(cfg.get('save_every',10))==0:
            save_checkpoint(out/'last.pt',controller,optimizers,final_update,cfg,elapsed)
        if final_update%int(cfg.get('snapshot_every',50))==0:
            save_checkpoint(out/f'update_{final_update:04d}.pt',controller,optimizers,final_update,cfg,elapsed)
        if elapsed>=cfg.get('max_seconds',1800):break
    save_checkpoint(out/'last.pt',controller,optimizers,final_update,cfg,time.perf_counter()-started)
    print('WARM_TRAIN_COMPLETE',str(out/'last.pt'),flush=True)
