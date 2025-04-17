    -- Declare the list of post IDs
    DECLARE @PostIds TABLE (PostId INT);
    INSERT INTO @PostIds (PostId)
    VALUES (2669573),(2460437),(1262415),(41353); -- all the ID's can be defined here

    -- Fetch the title and body of the questions and their highest voted answers
    SELECT 
        p.Id AS QuestionId,
        p.Title AS QuestionTitle,
        p.Body AS QuestionBody,
        a.Id AS AnswerId,
        a.Body AS AnswerBody,
        a.Score AS AnswerScore
    FROM 
        Posts p
    LEFT JOIN 
        (
            SELECT 
                a.ParentId,
                a.Id,
                a.Body,
                a.Score,
                ROW_NUMBER() OVER (PARTITION BY a.ParentId ORDER BY a.Score DESC) AS rn
            FROM 
                Posts a
            WHERE 
                a.ParentId IN (SELECT PostId FROM @PostIds)
                AND a.PostTypeId = 2
        ) a ON p.Id = a.ParentId
    WHERE 
        p.Id IN (SELECT PostId FROM @PostIds)
        AND p.PostTypeId = 1
        AND a.rn = 1;
