import Mathlib
set_option linter.style.longLine false
set_option linter.style.commandStart false

------------------------------------------
-- LAB SIX: THE FINAL BOSS
------------------------------------------

-- This lab will require you to recall everything you learned over the course of the semester by way of Lean techniques (as well as how to define and manipulate logical syntax).

-- Define an inductive type called pform, which obeys the rules of syntax of propositional logic formulas. A pform can be an atomic formula (call this constructor at -- which takes as a parameter any string, of type String in Lean), or built using existing pform objects and the constructors myNot, myAnd, myOr, and myImp (standing for the usual operators). You must specify the fact that there is an algorithm to test for the syntactic equality of two pform objects.
inductive pform : Type
  | at    : String → pform
  | myNot : pform → pform
  | myAnd : pform → pform → pform
  | myOr  : pform → pform → pform
  | myImp : pform → pform → pform
deriving DecidableEq

--------------------------------------------------------------------------------------

-- def a : pform := pform.myAnd (pform.at "P") (pform.myNot (pform.at "Q"))
-- def b : pform := pform.myAnd (pform.at "P") (pform.myNot (pform.at "Q"))
-- def c : pform := pform.myAnd (pform.myNot (pform.at "Q")) (pform.at "P")
-- #eval (a = b)
-- #eval (a = c)
-- #eval (a = pform.at "R")

--------------------------------------------------------------------------------------

-- Once you have defined this type, define an inductive object called pftree, which witnesses whether or not there is a proof of a pform φ from a finite set of pforms X (choose the type for this appropriately!) according to the rules given in Table 1 of https://www.cmi.ac.in/~spsuresh/pdfs/jlc2020-tr.pdf. This is a proof system in *intuitionistic* propositional logic, which means that it does not admit the law of excluded middle (i.e. φ ∨ ¬φ is not a tautology, and since this proof system is sound, cannot be proven without assumptions.) It also means, consequently, that anything proven must follow from the assumptions via a proof; there are no "free" axioms. Intuitionism has a long history, and goes back to Brouwer, and underpins much of theorem proving -- Lean's own underlying theory is intuitionistic.

open pform

inductive pftree : Finset pform → pform → Prop
  | ax {Γ α} :
      α ∈ Γ → pftree Γ α

  | ni {Γ α β} :
      pftree (Γ ∪ {α}) β →
      pftree (Γ ∪ {α}) (myNot β) →
      pftree Γ (myNot α)

  | ne {Γ α β} :
      pftree Γ β →
      pftree Γ (myNot β) →
      pftree Γ α

  | andi {Γ α β} :
      pftree Γ α →
      pftree Γ β →
      pftree Γ (myAnd α β)

  | ande_left {Γ α β} :
      pftree Γ (myAnd α β) →
      pftree Γ α

  | ande_right {Γ α β} :
      pftree Γ (myAnd α β) →
      pftree Γ β

  | ori_left {Γ α β} :
      pftree Γ α →
      pftree Γ (myOr α β)

  | ori_right {Γ α β} :
      pftree Γ β →
      pftree Γ (myOr α β)

  | ore {Γ α β γ} :
      pftree Γ (myOr α β) →
      pftree (Γ ∪ {α}) γ →
      pftree (Γ ∪ {β}) γ →
      pftree Γ γ

  | impi {Γ α β} :
      pftree (Γ ∪ {α}) β →
      pftree Γ (myImp α β)

  | impp {Γ α β} :
      pftree Γ β →
      pftree Γ (myImp α β)

  | impe {Γ α β} :
      pftree Γ (myImp α β) →
      pftree Γ α →
      pftree Γ β

-- In order to define a finite set, you will need to import Mathlib. Mathlib is THE Lean library, in that it includes a lot of handy constructs and tactics. In particular, it also includes the constructor called Finset, which takes as input a type, and spits out a finite set of said type (much like the List constructor). If you are using VSCode, if you type Finset followed by "." (exactly like with List) you can see the various methods and theorems you have access to under the Finset constructor. This is why we require the lines included above, which import the Mathlib library, and disable some irritating linters about line length and where a command should start.

-- Finally, define a theorem called mono_prf, which says that if there is a proof tree witnessing a proof of a pform φ from a Finset X of pforms, then there is a proof tree witnessing a proof of φ from a set which is a superset of X. Recall that you can state this in multiple ways; use whichever way seems most amenable to proving this statement. (This is what we have proved in class and called Monotonicity.) Submit your entire answer below this sequence of comments. In general, one wants to show not just Monotonicity, but various other desirable properties of any given proof system, including whether inference in them is decidable (and if it, how efficiently it can be done), like we are doing in that paper linked above.

theorem mono_prf {Γ Δ φ} (h : Γ ⊆ Δ) : pftree Γ φ → pftree Δ φ :=
  by
  intro pf
  induction pf generalizing Δ with
  | ax hmem =>
    exact pftree.ax (h hmem)

  | ni p1 p2 ih1 ih2 =>
    apply pftree.ni
    · apply ih1
      exact Finset.union_subset_union_left (α := pform) h
    · apply ih2
      exact Finset.union_subset_union_left (α := pform) h

  | ne p1 p2 ih1 ih2 =>
    exact pftree.ne (ih1 h) (ih2 h)

  | andi p1 p2 ih1 ih2 =>
    exact pftree.andi (ih1 h) (ih2 h)

  | ande_left p ih =>
    exact pftree.ande_left (ih h)

  | ande_right p ih =>
    exact pftree.ande_right (ih h)

  | ori_left p ih =>
    exact pftree.ori_left (ih h)

  | ori_right p ih =>
    exact pftree.ori_right (ih h)

  | ore p pα pβ ih ihα ihβ =>
    apply pftree.ore
    · exact ih h
    · apply ihα
      exact Finset.union_subset_union_left (α := pform) h
    · apply ihβ
      exact Finset.union_subset_union_left (α := pform) h

  | impi p ih =>
    apply pftree.impi
    apply ih
    exact Finset.union_subset_union_left (α := pform) h

  | impe p1 p2 ih1 ih2 =>
    exact pftree.impe (ih1 h) (ih2 h)

  | impp p ih =>
    apply pftree.impp
    exact ih h
