with transfer_counts as (
    select hadm_id, subject_id, count(transfer_id) as transfer_count
    from hosp.transfers
    group by hadm_id, subject_id
), 
max_transfer_count as (
    select max(transfer_count) as max_transfers
    from transfer_counts
), 
longest_chain_patients as (
    select tc.subject_id, tc.hadm_id, tc.transfer_count
    from transfer_counts tc
    join max_transfer_count mtc on tc.transfer_count = mtc.max_transfers
)
select t.subject_id, t.hadm_id, array_agg(t.transfer_id order by t.intime) as transfers  
from hosp.transfers t
join longest_chain_patients lcp on t.subject_id = lcp.subject_id
where t.hadm_id is not null
group by t.subject_id, t.hadm_id
order by array_length(array_agg(t.transfer_id order by t.intime), 1) asc, t.hadm_id;