
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
	b26 - block
	b27 - block
	b28 - block
	b29 - block
	b3 - block
	b30 - block
	b31 - block
	b32 - block
	b33 - block
	b34 - block
	b35 - block
	b36 - block
	b37 - block
	b38 - block
	b39 - block
	b4 - block
	b40 - block
	b41 - block
	b42 - block
	b43 - block
	b44 - block
	b45 - block
	b46 - block
	b47 - block
	b48 - block
	b49 - block
	b5 - block
	b50 - block
	b51 - block
	b52 - block
	b53 - block
	b54 - block
	b55 - block
	b56 - block
	b57 - block
	b58 - block
	b59 - block
	b6 - block
	b7 - block
	b8 - block
	b9 - block
  )
  (:init 
	(clear b0)
	(clear b11)
	(clear b13)
	(clear b15)
	(clear b17)
	(clear b18)
	(clear b19)
	(clear b20)
	(clear b21)
	(clear b23)
	(clear b25)
	(clear b27)
	(clear b29)
	(clear b2)
	(clear b31)
	(clear b32)
	(clear b33)
	(clear b34)
	(clear b36)
	(clear b37)
	(clear b38)
	(clear b3)
	(clear b40)
	(clear b42)
	(clear b43)
	(clear b45)
	(clear b46)
	(clear b48)
	(clear b4)
	(clear b50)
	(clear b52)
	(clear b53)
	(clear b55)
	(clear b56)
	(clear b57)
	(clear b59)
	(clear b5)
	(clear b6)
	(clear b7)
	(clear b9)
	(handempty )
	(on b0 b1)
	(on b11 b12)
	(on b13 b14)
	(on b15 b16)
	(on b21 b22)
	(on b23 b24)
	(on b25 b26)
	(on b27 b28)
	(on b29 b30)
	(on b34 b35)
	(on b38 b39)
	(on b40 b41)
	(on b43 b44)
	(on b46 b47)
	(on b48 b49)
	(on b50 b51)
	(on b53 b54)
	(on b57 b58)
	(on b7 b8)
	(on b9 b10)
	(ontable b10)
	(ontable b12)
	(ontable b14)
	(ontable b16)
	(ontable b17)
	(ontable b18)
	(ontable b19)
	(ontable b1)
	(ontable b20)
	(ontable b22)
	(ontable b24)
	(ontable b26)
	(ontable b28)
	(ontable b2)
	(ontable b30)
	(ontable b31)
	(ontable b32)
	(ontable b33)
	(ontable b35)
	(ontable b36)
	(ontable b37)
	(ontable b39)
	(ontable b3)
	(ontable b41)
	(ontable b42)
	(ontable b44)
	(ontable b45)
	(ontable b47)
	(ontable b49)
	(ontable b4)
	(ontable b51)
	(ontable b52)
	(ontable b54)
	(ontable b55)
	(ontable b56)
	(ontable b58)
	(ontable b59)
	(ontable b5)
	(ontable b6)
	(ontable b8)
  )
  (:goal (and
	(on b39 b36)
	(on b36 b35)
	(ontable b35)))
)
