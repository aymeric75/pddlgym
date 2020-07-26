
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
	(clear b18)
	(clear b19)
	(clear b2)
	(clear b4)
	(clear b7)
	(handempty )
	(on b0 b1)
	(on b10 b11)
	(on b12 b13)
	(on b14 b15)
	(on b16 b17)
	(on b19 b20)
	(on b2 b3)
	(on b4 b5)
	(on b5 b6)
	(on b7 b8)
	(on b8 b9)
	(ontable b11)
	(ontable b13)
	(ontable b15)
	(ontable b17)
	(ontable b18)
	(ontable b1)
	(ontable b20)
	(ontable b3)
	(ontable b6)
	(ontable b9)
  )
  (:goal (and
	(on b20 b9)
	(on b9 b14)
	(on b14 b8)
	(on b8 b7)
	(on b7 b11)
	(ontable b11)))
)
