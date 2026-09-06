-- LECTURES
create table if not exists public.lectures (
    lecture_id      uuid primary key default gen_random_uuid(),
    user_id         uuid not null references auth.users (id) on delete cascade,
    title           text,
    uploaded_files  jsonb,
    combined_text   text,
    summary         text,
    created_at      timestamptz not null default now()
);

create index if not exists lectures_user_created_idx on public.lectures (user_id, created_at desc);

-- RESULTS
create table if not exists public.results (
    result_id       uuid primary key default gen_random_uuid(),
    lecture_id      uuid not null references public.lectures (lecture_id) on delete cascade,
    user_id         uuid not null references auth.users (id) on delete cascade,
    quiz_questions  jsonb,
    user_answers    jsonb,
    performance     jsonb,
    feedback        text,
    study_plan      jsonb,
    completed_at    timestamptz not null default now()
);

create index if not exists results_lecture_idx on public.results (lecture_id, completed_at);

-- ROW LEVEL SECURITY
alter table public.lectures enable row level security;
alter table public.results enable row level security;

drop policy if exists "own lectures" on public.lectures;
create policy "own lectures" on public.lectures for all using (auth.uid() = user_id) with check (auth.uid() = user_id);

drop policy if exists "own results" on public.results;
create policy "own results" on public.results for all using (auth.uid() = user_id) with check (auth.uid() = user_id);
