(define (problem blocks)
    (:domain glibblocks)
    (:objects 
        d - block
        b - block
        a - block
        c - block
        e - block
        f - block
        robot - robot
    )
    (:init
        (clear a)
        (ontable a)
        (clear b)
        (ontable b)
        (clear c)
        (ontable c)
        (clear d)
        (ontable d)
        (clear e)
        (ontable e)
        (clear f)
        (ontable f)
        (handempty robot)

        ; action literals
        (pickup a)
        (putdown a)
        (unstack a)
        (stack a b)
        (stack a c)
        (stack a d)
        (stack a e)
        (stack a f)
        (pickup b)
        (putdown b)
        (unstack b)
        (stack b a)
        (stack b c)
        (stack b d)
        (stack b e)
        (stack b f)
        (pickup c)
        (putdown c)
        (unstack c)
        (stack c b)
        (stack c a)
        (stack c d)
        (stack c e)
        (stack c f)
        (pickup d)
        (putdown d)
        (unstack d)
        (stack d b)
        (stack d c)
        (stack d a)
        (stack d e)
        (stack d f)
        (pickup e)
        (putdown e)
        (unstack e)
        (stack e b)
        (stack e c)
        (stack e a)
        (stack e d)
        (stack e f)
        (pickup f)
        (putdown f)
        (unstack f)
        (stack f b)
        (stack f c)
        (stack f a)
        (stack f d)
        (stack f e)

    )
    (:goal (and (on a b) (on b c) (on d e) (on e f)))
)
