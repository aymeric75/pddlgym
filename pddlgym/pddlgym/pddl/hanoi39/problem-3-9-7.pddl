(define (problem hanoi1)
  (:domain hanoi)
  (:objects peg1 peg2 peg3 peg4 peg5 peg6 peg7 peg8 peg9 d1 d2 d3)
  (:init
   (smaller peg1 d1) (smaller peg1 d2) (smaller peg1 d3) 
   (smaller peg2 d1) (smaller peg2 d2) (smaller peg2 d3)
   (smaller peg3 d1) (smaller peg3 d2) (smaller peg3 d3)
   (smaller peg4 d1) (smaller peg4 d2) (smaller peg4 d3)
   (smaller peg5 d1) (smaller peg5 d2) (smaller peg5 d3) 
   (smaller peg6 d1) (smaller peg6 d2) (smaller peg6 d3)
   (smaller peg7 d1) (smaller peg7 d2) (smaller peg7 d3)
   (smaller peg8 d1) (smaller peg8 d2) (smaller peg8 d3)
   (smaller peg9 d1) (smaller peg9 d2) (smaller peg9 d3) 


   (smaller d2 d1) (smaller d3 d1) (smaller d3 d2)
   

   (clear peg1) (clear peg2) (clear peg3) (clear peg4) (clear peg5) (clear peg6) (clear peg7) (clear peg9) (clear d1)
   (on d3 peg8) (on d2 d3) (on d1 d2)
  )
  (:goal (and (on d2 d3) (on d3 d2)))
  )