; Probabilistic triangle tireworld.

(define (domain tireworld)
  (:requirements :typing :strips :probabilistic-effects)
  (:types location)
  (:predicates
       (vehicle-at ?loc - location)
       (spare-in ?loc - location)
       (road ?from - location ?to - location)
       (not-flattire)
       (movecar ?to - location)
       (changetire ?loc - location)
  )

  ; (:actions movecar changetire)

  (:action move-car
    :parameters (?from - location ?to - location)
    :precondition (and (vehicle-at ?from) (road ?from ?to) (not-flattire) (movecar ?to))
    :effect (and (vehicle-at ?to) (not (vehicle-at ?from))
       (probabilistic 0.5 (not (not-flattire)))))
  (:action changetire
    :parameters (?loc - location)
    :precondition (and (spare-in ?loc) (vehicle-at ?loc) (changetire ?loc))
    :effect (and (not (spare-in ?loc)) (not-flattire)))
)
