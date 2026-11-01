How automatic searches are sized
================================

Leave the population size, the number of generations, or the basin-hopping
hop count on ``"auto"`` and SCGO sizes the search from how many distinct
shapes the system has. That count is a lattice animal: shapes grow as
:math:`\mu^N`, and two shapes that differ only by a rotation are one shape.
A growth factor tempers the count, because a search does not have to try
every shape. Extra slots widen the population. The number of generations
follows how many atoms are moving, until a population would pass 1000.
Neither number goes above 1000. The hop count does not either.

What makes a search harder
--------------------------

A bigger cluster has more shapes, so it gets more slots. The number of
generations stays on the atom-count curve below. The population takes the
extra slots.

Mixing several chemical elements multiplies the arrangements, so a
platinum-cobalt cluster gets a wider population than a pure cluster with the
same number of atoms. The number of generations stays the same.

A supported cluster can be turned only in the plane of the surface. A vacuum
cluster can be turned in any direction, and those turns are the same shape.
The cluster is still three-dimensional, so it uses the same growth constant.
Only the rotation group changes, and a supported cluster asks for slightly
more slots than the vacuum cluster of the same size. The surface underneath
is held still and is not counted. The budget does not go below the count set
by how many atoms are moving, so a small supported search barely widens.

The atoms of a molecule are allowed to move, so they raise that atom-count
floor. Where the molecule sits is a separate placement term. One molecule
changes the slot count only a little. Several molecules on the same cluster
make the population wider.

Equations
---------

Natural logarithms throughout. For a cluster of :math:`N` core atoms with
counts :math:`n_1,\ldots,n_M`,

.. math::

   \log \Omega = N \log \mu - \log \lvert G \rvert
   + \log \frac{N!}{n_1! \cdots n_M!}
   + \log \Omega_{\mathrm{sites}}.

:math:`\mu` is 8.35, the growth constant of site animals on the cubic lattice.
The series estimate is 8.347. A supported cluster uses the same constant,
because it is still a three-dimensional island. :math:`\lvert G \rvert` is 24
in vacuum and 4 on a surface: the proper rotations of the cube, and the
proper rotations about the surface normal. A hexagonal surface would use six
rotations. That changes the slot count by about 1%, so every surface uses
four. Counting reflections as well (48 and 8) leaves the ratio unchanged. A
unary supported cluster and the vacuum cluster of the same size then differ
only by that symmetry, and the slot ratio is :math:`6^{\alpha}`. Identical
molecules of :math:`k` fragments, with :math:`f_i` copies of each kind, add

.. math::

   \log \Omega_{\mathrm{sites}}
   = \log \frac{n_{\mathrm{sites}}!}{(n_{\mathrm{sites}} - k)! \, f_1! \cdots f_m!}.

:math:`n_{\mathrm{sites}}` is :math:`\mathrm{round}(N^{2/3})` for a vacuum
cluster and for a supported island, and at least :math:`k`. That count follows
the exposed area. Surface atoms are a few times :math:`N^{2/3}`, and a
constant factor of that kind enters the budget only through
:math:`\Omega^{\alpha}`, so the prefactor is left out. On a bare slab there
is no core term. :math:`n_{\mathrm{sites}}` is then the number of slab atoms
allowed to move.

The anchor :math:`\Omega_{\mathrm{ref}}` is a unary vacuum cluster, including
when the system itself is supported. For a genetic algorithm that anchor has
10 atoms and :math:`E_{\mathrm{ref}}` is the square of the atom-count budget
at 10 atoms, :math:`86 \times 86`. The slot count is

.. math::

   \log E = \log E_{\mathrm{ref}}
   + \alpha \, (\log \Omega - \log \Omega_{\mathrm{ref}}),

with growth factor :math:`\alpha = 0.03`. The same-size comparison would
cancel :math:`N \log \mu`. The anchor is a fixed cluster so that growth stays
in the budget.

Let :math:`N_{\mathrm{mobile}}` be the number of atoms allowed to move, at
least 1. Generations, and the minimum population, follow

.. math::

   M = \min\bigl(1000,\, \max\bigl(3,\,
   \lfloor b + s \ln(1 + N_{\mathrm{mobile}}) \rfloor\bigr)\bigr),

with offset :math:`b = 3` and slope :math:`s = 35`. Near 10 atoms,
:math:`M^2` grows by about 7% per added atom, and the animal budget grows by
:math:`\alpha \ln \mu`, about 6.4%. Through the low teens the two stay
together, and this floor is the generation count. The anchor product is the
floor at 10 atoms. Platinum first-hit counts from 4 to 11 atoms run from 1
to 232 relaxations, and :math:`86 \times 86` is about thirty times the
longest of those counts. A genetic algorithm spends extra slots on the
population. Generations stay at :math:`M` until the population would pass
1000:

.. math::

   (P, G) =
   \begin{cases}
   (M, M) & \text{if } E \le M^{2}, \\
   \bigl(\min(1000, \lceil E/M \rceil),\; M\bigr)
     & \text{if that product covers } E, \\
   \bigl(1000,\; \min(1000, \lceil E/1000 \rceil)\bigr)
     & \text{otherwise.}
   \end{cases}

Both numbers stay inside 3 and 1000. A pure 10-atom vacuum cluster is the
anchor, so its pair is 86 by 86. The same cluster on a surface is 91 by 86.
Pt38 is 336 by 131. Pt55 is 908 by 143, and on a surface it is 958 by 143.
Pt5Co5 is 102 by 86. Basin hopping of a cluster uses the same :math:`\alpha`
and the same :math:`\Omega`, anchored on the atom-count hop count of a
3-atom cluster. That hop count is
:math:`\min(1000, \max(M, \lceil E \rceil))`.

A bare slab with no molecules uses :math:`M` for the population, the
generations, and the hop count. Molecules on that slab add
:math:`\alpha \log \Omega_{\mathrm{sites}}` on top of :math:`\log(M^2)` for
the genetic-algorithm slot count, so
:math:`E = M^2 \Omega_{\mathrm{sites}}^{\alpha}`. Basin hopping on the same
slab grows from the hop count, not from that square:

.. math::

   \mathrm{hops} = \min\bigl(1000,\,
   \lceil M \,\Omega_{\mathrm{sites}}^{\alpha} \rceil\bigr).

A bare slab has :math:`\Omega_{\mathrm{sites}} = 1`, so the hop count stays
at :math:`M`. The site term widens the genetic-algorithm population through
:math:`E` and scales the hop count only by
:math:`\Omega_{\mathrm{sites}}^{\alpha}`.

Basin hopping
-------------

Basin hopping of a cluster uses the animal count for its number of hops.
SCGO picks basin hopping for three-atom clusters with no molecule attached,
and those runs use the hop count set by that atom count. A larger cluster
search, including one chosen by hand, gets more hops from the same growth
factor, never fewer than the atom-count hop count for its own size, and at
most 1000. The Lennard-Jones waits at 75 and 102 atoms sit above that cap.
A slab uses the formula above: :math:`M` with no molecules, and :math:`M`
times :math:`\Omega_{\mathrm{sites}}^{\alpha}` when molecules are present.

One optimization, not a search
------------------------------

The number of steps inside one local relaxation, and the number of steps in a
nudged-elastic-band calculation, follow how many atoms are moving. Those
limits belong to a single optimization, not to how many different structures
to try.

A search of the top layers of a bare slab uses that same atom-count budget.
The count uses the atoms that are allowed to move, not an empty formula.

How the growth factor was chosen
--------------------------------

:math:`\alpha` is one slope for every system. It does not rank an awkward
cluster above an easier, larger one. Pt5 and Pt7 took longer to find than
Pt8 through Pt11, and a 55-atom Lennard-Jones cluster is easier than a
38-atom one. Setting the factor to zero because of that scatter removes
:math:`\mu^N` from the budget. The factor is the average growth, not a
classifier of hard cases.

Two sets fix the slope, with a separate intercept for each set because the
efforts are not in the same units. Neither set is shipped with this package.

The first set is local MACE searches on one GPU, three seeds (42, 43, and
44). Each pure platinum cluster from 4 to 11 atoms found the reference
geometry. The effort is the geometric mean of the relaxations until that
first hit. On their own, those eight points give a slope of −0.06 with a
standard error of 0.14 and :math:`R^2` 0.03. They are consistent with the
slope below, and they do not resolve a different one. Pt4Co4, same three
seeds, first reached the shared best energy after 19, 87, and 160
relaxations. That spread sits inside the pure clusters (1 to 232
relaxations), so mixing keeps the same factor. Supported clusters were not
part of the fit.

The second set is published basin-hopping searches of Lennard-Jones
clusters. The waiting times are about 1000 steps for 38 atoms, :math:`1.2
\times 10^5` for 75 atoms (4 successes in 100 runs of length 5000), and
:math:`1.6 \times 10^5` for 102 atoms (3 successes in 100 runs of the same
length). A 55-atom cluster was found in under 200 steps. That figure is an
upper bound, and it is easier than the 38-atom cluster, so it is not a point
in the fit.

One point per cluster, a shared slope, and a separate intercept for each
set, gives :math:`\alpha = 0.037` with a standard error of 0.019. Rounded
to two digits, the factor is 0.03. The Lennard-Jones points alone give
0.03 as well. Using every platinum seed instead of the geometric mean, or
the older 38-atom waiting time of about :math:`3 \times 10^3`, stays inside
that uncertainty.
