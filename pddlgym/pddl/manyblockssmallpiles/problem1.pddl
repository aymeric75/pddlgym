
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
	(clear b11)
	(clear b12)
	(clear b13)
	(clear b14)
	(clear b16)
	(clear b18)
	(clear b1)
	(clear b21)
	(clear b4)
	(clear b7)
	(clear b8)
	(handempty )
	(on b14 b15)
	(on b16 b17)
	(on b18 b19)
	(on b19 b20)
	(on b1 b2)
	(on b2 b3)
	(on b4 b5)
	(on b5 b6)
	(on b8 b9)
	(on b9 b10)
	(ontable b0)
	(ontable b10)
	(ontable b11)
	(ontable b12)
	(ontable b13)
	(ontable b15)
	(ontable b17)
	(ontable b20)
	(ontable b21)
	(ontable b3)
	(ontable b6)
	(ontable b7)
  )
  (:goal (and
	(on b6 b16)
	(on b16 b20)
	(on b20 b4)
	(ontable b4)))
)
