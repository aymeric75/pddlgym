


(define (problem logistics-c5-s1-p4-a2)
(:domain logistics-strips)
(:objects a0 a1 
          c0 c1 c2 c3 c4 
          t0 t1 t2 t3 t4 
          l00 l10 l20 l30 l40 
          p0 p1 p2 p3 
)
(:init
(AIRPLANE a0)
(AIRPLANE a1)
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
(at t0 l00)
(at t1 l10)
(at t2 l20)
(at t3 l30)
(at t4 l40)
(at p0 l00)
(at p1 l30)
(at p2 l00)
(at p3 l10)
(at a0 l00)
(at a1 l10)
)
(:goal
(and
(at p0 l10)
(at p1 l30)
(at p2 l30)
(at p3 l30)
)
)
)


