SELECT distinct 
    hcpcs_cd, short_description
FROM 
    hosp.hcpcsevents
WHERE 
    lower(short_description) LIKE '%hospital observation%'
ORDER BY 
    hcpcs_cd, short_description;
