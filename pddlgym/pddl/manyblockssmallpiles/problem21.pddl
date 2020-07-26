
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
	(clear b13)
	(clear b14)
	(clear b15)
	(clear b17)
	(clear b19)
	(clear b3)
	(clear b6)
	(clear b9)
	(handempty )
	(on b0 b1)
	(on b10 b11)
	(on b11 b12)
	(on b15 b16)
	(on b17 b18)
	(on b19 b20)
	(on b1 b2)
	(on b20 b21)
	(on b3 b4)
	(on b4 b5)
	(on b6 b7)
	(on b7 b8)
	(ontable b12)
	(ontable b13)
	(ontable b14)
	(ontable b16)
	(ontable b18)
	(ontable b21)
	(ontable b2)
	(ontable b5)
	(ontable b8)
	(ontable b9)
  )
  (:goal (and
	(on b0 b17)
	(on b17 b19)
	(ontable b19)
	(on b16 b6)
	(on b6 b12)
	(on b12 b2)
	(on b2 b20)
	(on b20 b1)
	(ontable b1)))
)
