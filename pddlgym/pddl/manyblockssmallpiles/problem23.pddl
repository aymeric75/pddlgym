
(define (problem manyblockssmallpiles) (:domain blocks)
  (:objects
        b0 - block
	b1 - block
	b10 - block
	b11 - block
	b2 - block
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
	(clear b1)
	(clear b3)
	(clear b4)
	(clear b5)
	(clear b7)
	(clear b8)
	(handempty )
	(on b1 b2)
	(on b5 b6)
	(on b8 b9)
	(on b9 b10)
	(ontable b0)
	(ontable b10)
	(ontable b11)
	(ontable b2)
	(ontable b3)
	(ontable b4)
	(ontable b6)
	(ontable b7)
  )
  (:goal (and
	(on b11 b5)
	(on b5 b10)
	(on b10 b9)
	(ontable b9)))
)
