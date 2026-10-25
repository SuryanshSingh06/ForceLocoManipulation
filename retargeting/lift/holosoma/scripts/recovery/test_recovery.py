import unittest
from types import SimpleNamespace
from t2_recover_holosoma_lifts import recovery_solve
from t2_recover_holosoma_lifts import checked_iterate
from unittest.mock import patch
import numpy as np


class RecoveryTests(unittest.TestCase):
    def test_default_success_unchanged(self):
        solver=SimpleNamespace(step_size=.2,foot_sticking_tolerance=.001)
        events=[]
        result=recovery_solve(lambda *a,**k: 'ok',solver,(),{},events)
        self.assertEqual(result,'ok')
        self.assertEqual(events,[])

    def test_bounded_retry_restores_settings(self):
        solver=SimpleNamespace(step_size=.2,foot_sticking_tolerance=.001)
        calls=[];events=[]
        def solve(s,**kwargs):
            calls.append((s.step_size,s.foot_sticking_tolerance))
            if s.foot_sticking_tolerance < .005:
                raise RuntimeError('CVXPY solve failed: infeasible')
            return 'ok'
        self.assertEqual(recovery_solve(solve,solver,(),dict(frame_idx=19),events),'ok')
        self.assertEqual(calls[-1],(.8,.005))
        self.assertEqual(events[-1]['frame'],19)
        self.assertEqual((solver.step_size,solver.foot_sticking_tolerance),(.2,.001))

    def test_exhaustion_never_fakes_success(self):
        solver=SimpleNamespace(step_size=.2,foot_sticking_tolerance=.001)
        calls=[]
        def solve(s,**kwargs):
            calls.append((s.step_size,s.foot_sticking_tolerance))
            raise RuntimeError('CVXPY solve failed: infeasible')
        with self.assertRaisesRegex(RuntimeError,'exhausted'):
            recovery_solve(solve,solver,(),{},[])
        self.assertEqual(len(calls),6)
        self.assertLessEqual(max(c[0] for c in calls),.8)
        self.assertLessEqual(max(c[1] for c in calls),.01)
        self.assertEqual(solver.step_size,.2)

    def test_unrelated_error_not_retried(self):
        solver=SimpleNamespace(step_size=.2,foot_sticking_tolerance=.001)
        calls=[]
        def solve(s,**kwargs):
            calls.append(1)
            raise RuntimeError('missing model')
        with self.assertRaisesRegex(RuntimeError,'missing model'):
            recovery_solve(solve,solver,(),{},[])
        self.assertEqual(calls,[1])


class ContinuationTests(unittest.TestCase):
    def test_continuation_preserves_endpoint_and_foot_anchor(self):
        start=np.zeros(43);start[3]=start[-4]=1.
        target=start.copy();target[-7]=1.
        solver=SimpleNamespace(has_dynamic_object=True)
        anchors=[];calls=[];events=[]
        def iterate(self,q_locked,q_n,q_t_last,target_laplacian=None,
                    adj_list=None,obj_pts_local=None,foot_sticking=None,
                    init_t=False,n_iter=10,frame_idx=11):
            calls.append(1)
            anchors.append(q_t_last.copy())
            if abs(q_locked[-7]-q_n[-7])>.3:
                raise RuntimeError('Bounded recovery exhausted at frame 11')
            return q_locked.copy(),1.
        with patch('t2_recover_holosoma_lifts.constraint_check',return_value=dict(collision_excess_m=0.,joint_limit_excess=0.)):
            q,cost=checked_iterate(iterate,solver,(),dict(q_locked=target,q_n=start,q_t_last=start),events)
        np.testing.assert_array_equal(q[-7:],target[-7:])
        self.assertEqual(len(calls),5)
        self.assertEqual(events[-1]['subdivisions'],4)
        for anchor in anchors:np.testing.assert_array_equal(anchor,start)

    def test_nonlinear_collision_not_accepted(self):
        start=np.zeros(43);start[3]=start[-4]=1.
        target=start.copy();target[-7]=1.
        solver=SimpleNamespace(has_dynamic_object=True)
        def iterate(self,q_locked,q_n,q_t_last,init_t=False,n_iter=10,frame_idx=11):
            return q_locked.copy(),1.
        with patch('t2_recover_holosoma_lifts.constraint_check',return_value=dict(collision_excess_m=.01,joint_limit_excess=0.)):
            with self.assertRaisesRegex(RuntimeError,'continuation exhausted'):
                checked_iterate(iterate,solver,(),dict(q_locked=target,q_n=start,q_t_last=start),[])


if __name__=='__main__':unittest.main()
