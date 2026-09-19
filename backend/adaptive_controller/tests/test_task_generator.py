from backend.adaptive_controller.task_generator import generate_task_pool, split_tasks


def test_generates_requested_count():
    tasks = generate_task_pool(n_per_difficulty=10)
    assert len(tasks) == 30


def test_reproducible_with_same_seed():
    a = generate_task_pool(n_per_difficulty=5, seed=1)
    b = generate_task_pool(n_per_difficulty=5, seed=1)
    assert [t.task_id for t in a] == [t.task_id for t in b]
    assert [s.bytes for s in a[0].steps] == [s.bytes for s in b[0].steps]


def test_different_seeds_differ():
    a = generate_task_pool(n_per_difficulty=5, seed=1)
    b = generate_task_pool(n_per_difficulty=5, seed=2)
    assert [s.bytes for s in a[0].steps] != [s.bytes for s in b[0].steps]


def test_split_has_no_overlap():
    tasks = generate_task_pool(n_per_difficulty=20)
    train, val, test = split_tasks(tasks)
    train_ids = {t.task_id for t in train}
    val_ids = {t.task_id for t in val}
    test_ids = {t.task_id for t in test}
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)


def test_split_covers_all_tasks():
    tasks = generate_task_pool(n_per_difficulty=20)
    train, val, test = split_tasks(tasks)
    assert len(train) + len(val) + len(test) == len(tasks)


def test_split_is_stratified_by_difficulty():
    tasks = generate_task_pool(n_per_difficulty=20)
    train, val, test = split_tasks(tasks)
    for split in (train, val, test):
        difficulties = {t.difficulty for t in split}
        assert difficulties == {"easy", "medium", "hard"}