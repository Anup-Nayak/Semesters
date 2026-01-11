create or replace view ordered_nodes as(
 select  subject_id, hadm_id, admittime, dischtime 
    from hosp.admissions 
    order by admittime 
    limit 200
    
);
create or replace view nodes as( 
    select distinct ordered_nodes.subject_id, ordered_nodes.hadm_id, diagnoses_icd.icd_code, diagnoses_icd.icd_version, ordered_nodes.admittime, ordered_nodes.dischtime
    from hosp.diagnoses_icd
	join ordered_nodes  on diagnoses_icd.hadm_id = ordered_nodes.hadm_id and diagnoses_icd.subject_id = ordered_nodes.subject_id
    where ordered_nodes.dischtime is not null and ordered_nodes.admittime is not null and ordered_nodes.admittime<=ordered_nodes.dischtime
); --
 create or replace view edge_list(subject_id1, subject_id2) as( 
    select distinct n1.subject_id , n2.subject_id  from nodes n1 join nodes n2 
    on n1.subject_id != n2.subject_id 
    where ((n1.admittime <= n2.admittime and n2.admittime <= n1.dischtime )
    or (n2.admittime <= n1.admittime and n1.admittime <= n2.dischtime) )    
    and n1.icd_code = n2.icd_code and n1.icd_version = n2.icd_version
	AND n1.subject_id < n2.subject_id
	order by n1.subject_id asc,n2.subject_id
);


select * from edge_list