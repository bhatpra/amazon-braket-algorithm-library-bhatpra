# Copyright Amazon.com Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

from braket.devices import LocalSimulator

from braket.experimental.algorithms.chsh_inequality import (
    create_chsh_inequality_circuits,
    get_chsh_results,
    run_chsh_inequality,
)


def create_chsh_inequality_circuits():
    circuits = create_chsh_inequality_circuits()
    assert len(circuits) == 4


def test_run_chsh_inequality():
    circuits = create_chsh_inequality_circuits()
    local_tasks = run_chsh_inequality(circuits, LocalSimulator(), shots=0)
    assert len(local_tasks) == 4


def test_get_chsh_results():
    circuits = create_chsh_inequality_circuits()
    local_tasks = run_chsh_inequality(circuits, LocalSimulator(), shots=0)
    chsh_value, results, E_ab, E_ab_, E_a_b, E_a_b_ = get_chsh_results(local_tasks)
    assert len(results) == 4
