create or replace view temp2 as(
 select  subject_id, hadm_id, admittime, dischtime 
    from hosp.admissions 
    order by admittime 
    limit 200
);

create or replace view edge_list1(subject_id1,subject_id2) as( 
    select distinct n1.subject_id  , n2.subject_id   
	from temp2 n1 join temp2 n2 
    on n1.subject_id != n2.subject_id 
    where ((n1.admittime <= n2.admittime and n2.admittime <= n1.dischtime )
    or (n2.admittime <= n1.admittime and n1.admittime <= n2.dischtime) )    
   
	 -- Add condition to enforce that subject_id1 < subject_id2
   
	order by n1.subject_id asc
);
WITH RECURSIVE shortest_paths AS (
    -- Base case: Start from the given subject_id
    SELECT 
        subject_id1 AS start_id, 
        subject_id2 AS connected_id, 
        1 AS path_length,
        --ARRAY[subject_id1 || '-' || subject_id2] AS visited_edges -- Track visited edges
		 ARRAY[subject_id1 || '-' || subject_id2] AS visited_edges
    FROM edge_list1
    WHERE subject_id1 = 10038081

    UNION ALL

    -- Recursive case: Find all connected nodes, ensuring edges are not revisited
    SELECT 
        sp.start_id, 
        e.subject_id2 AS connected_id, 
        sp.path_length + 1,
        sp.visited_edges || (sp.connected_id || '-' || e.subject_id2) -- Append the current edge to the visited list
    FROM shortest_paths sp
    JOIN edge_list1 e 
        ON sp.connected_id = e.subject_id1
		
   WHERE (NOT (sp.connected_id || '-' || e.subject_id2) = ANY (sp.visited_edges))  and sp.start_id <> e.subject_id2 -- Skip revisited edges
)
select (SELECT  path_length
FROM (
    SELECT 
        start_id, 
        connected_id, 
        path_length, 
        ROW_NUMBER() OVER (PARTITION BY start_id, connected_id ORDER BY path_length ASC) AS rn
    FROM shortest_paths
	WHERE connected_id = 10021487
) ranked_paths
WHERE rn = 1
order by path_length asc , connected_id asc

)