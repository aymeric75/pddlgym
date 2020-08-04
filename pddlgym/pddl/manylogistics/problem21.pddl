


(define (problem logistics-c5-s1-p5-a6)
(:domain logistics-strips)
(:objects a0 a1 a2 a3 a4 a5 
          c0 c1 c2 c3 c4 
          t0 t1 t2 t3 t4 
          l00 l10 l20 l30 l40 
          p0 p1 p2 p3 p4 
)
(:init
(AIRPLANE a0)
(AIRPLANE a1)
(AIRPLANE a2)
(AIRPLANE a3)
(AIRPLANE a4)
(AIRPLANE a5)
(CITY c0)
(CITY c1)
(CITY c2)
(CITY c3)
(CITY c4)
(TRUCK t0)
(TRUCK t1)
(TRUCK t2)
(TRUCK t3)
(TRUCK t4)
(LOCATION l00)
(in-city  l00 c0)
(LOCATION l10)
(in-city  l10 c1)
(LOCATION l20)
(in-city  l20 c2)
(LOCATION l30)
(in-city  l30 c3)
(LOCATION l40)
(in-city  l40 c4)
(AIRPORT l00)
(AIRPORT l10)
(AIRPORT l20)
(AIRPORT l30)
(AIRPORT l40)
(OBJ p0)
(OBJ p1)
(OBJ p2)
(OBJ p3)
(OBJ p4)
(at t0 l00)
(at t1 l10)
(at t2 l20)
(at t3 l30)
(at t4 l40)
(at p0 l20)
(at p1 l10)
(at p2 l20)
(at p3 l00)
(at p4 l20)
(at a0 l40)
(at a1 l40)
(at a2 l20)
(at a3 l30)
(at a4 l40)
(at a5 l30)
)
(:goal
(and
(at p0 l30)
(at p1 l00)
(at p2 l10)
(at p3 l00)
(at p4 l40)
)
)
)


