
(define (problem manyblockssmallpiles) (:domain blocks)
  (:objects
        b0 - block
	b1 - block
	b10 - block
	b11 - block
	b12 - block
	b13 - block
	b14 - block
	b15 - block
	b16 - block
	b17 - block
	b18 - block
	b19 - block
	b2 - block
	b20 - block
	b21 - block
	b22 - block
	b23 - block
	b24 - block
	b25 - block
	b3 - block
	b4 - block
	b5 - block
	b6 - block
	b7 - block
	b8 - block
	b9 - block
  )
  (:init 
	(clear b0)
	(clear b10)
	(clear b12)
	(clear b14)
	(clear b16)
	(clear b17)
	(clear b18)
	(clear b19)
	(clear b1)
	(clear b20)
	(clear b23)
	(clear b2)
	(clear b5)
	(clear b8)
	(handempty )
	(on b10 b11)
	(on b12 b13)
	(on b14 b15)
	(on b20 b21)
	(on b21 b22)
	(on b23 b24)
	(on b24 b25)
	(on b2 b3)
	(on b3 b4)
	(on b5 b6)
	(on b6 b7)
	(on b8 b9)
	(ontable b0)
	(ontable b11)
	(ontable b13)
	(ontable b15)
	(ontable b16)
	(ontable b17)
	(ontable b18)
	(ontable b19)
	(ontable b1)
	(ontable b22)
	(ontable b25)
	(ontable b4)
	(ontable b7)
	(ontable b9)
  )
  (:goal (and
	(on b19 b22)
	(on b22 b21)
	(on b21 b20)
	(ontable b20)))
)
