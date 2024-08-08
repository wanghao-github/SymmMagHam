      program read_magnetic_data
* read data about magnetic space groups
* input data from magnetic_table.dat
      implicit none
      character(len=20) filename
      integer i,j,k,m,n
      
* for the ith nonhexagonal point operator:
* point_op_label(i): point operator symbol (from Litvin)
      character point_op_label(48)*8
* point_op_xyz(i): point operator in x,y,z notation
      character point_op_xyz(48)*10
* point_op_matrix(i): point operator matrix
      integer point_op_matrix(3,3,48)

* for the ith hexagonal point operator:
* point_op_hex_label(i): point operator symbol (from Litvin)
      character point_op_hex_label(24)*8
* point_op_hex_xyz(i): point operator in x,y,z notation
      character point_op_hex_xyz(24)*10
* point_op_hex_matrix(i): point operator matrix
      integer point_op_hex_matrix(3,3,24)

* number of magnetic space groups
      integer magcount
      parameter(magcount=1651)

* for the ith magnetic space group
* nlabel_bns(i): numerical label in BNS setting
      character nlabel_bns(magcount)*12
* nlabel_parts_bns(j,i): jth part of nlabel_bns
      integer nlabelparts_bns(2,magcount)
* spacegroup_label_uinfied(i): unified group symbol
      character spacegroup_label_unified(magcount)*27
* spacegroup_label_bns(i): group symbol
      character spacegroup_label_bns(magcount)*14
* nlabel_og(i): numerical label in OG setting
      character nlabel_og(magcount)*12
* nlabel_parts_og(j,i): jth part of nlabel_og
      integer nlabelparts_og(3,magcount)
* label_og(i): group symbol
      character spacegroup_label_og(magcount)*14
* magtype(i): type of magnetic space group (1-4)
      integer magtype(magcount)

* BNS-OG transformation (if type-4)
* bnsog_point_op(j,k,i): 3x3 point operator part of transformation
      integer bnsog_point_op(3,3,magcount)
* bnsog_origin(j,i): translation part of transformation
* bnsog_point_origin(i): common denominator
      integer bnsog_origin(3,magcount)
      integer bnsog_origin_denom(magcount)

* iops_count(i): number of point operators
      integer ops_count(magcount)
* wyckoff_count(i): number of wyckoff sites
      integer wyckoff_site_count(magcount)
* wyckoff_pos_count(j,i): number of positions in jth wyckoff site
      integer wyckoff_pos_count(27,magcount)
* wyckoff_mult(j,i): multiplicity for jth wyckoff site
      integer wyckoff_mult(27,magcount)
* wyckoff_label(j,i): symbol (a,b,c,...,z,alpha) for jth wyckoff site
      character wyckoff_label(27,magcount)

* for BNS setting
* lattice_bns_vectors_count(i): number of lattice vectors defining the lattice
      integer lattice_bns_vectors_count(magcount)
* lattice_bns_vectors(k,j,i): kth component of the jth lattice vector
* lattice_bns_vectors_denom(j,i): common denominator
      integer lattice_bns_vectors(3,6,magcount)
      integer lattice_bns_vectors_denom(6,magcount)
* for jth operator
* ops_bns_point_op(j,i): point operator part
      integer ops_bns_point_op(96,magcount)
* ops_bns_trans(k,j,i): kth component of translation part
* ops_bns_trans_denom(j,i): common denominator
      integer ops_bns_trans(3,96,magcount)
      integer ops_bns_trans_denom(96,magcount)
* ops_bns_timeinv(j,i): 1=no time inversion, -1=time inversion
      integer ops_bns_timeinv(96,magcount)
* for jth wyckoff site
* wyckoff_bns_fract(k,j,i): kth component of fractional part of wyckoff position
* wyckoff_bns_fract_denom(j,i): common denominator
      integer wyckoff_bns_fract(3,96,27,magcount)
      integer wyckoff_bns_fract_denom(96,27,magcount)
* wyckoff_bns_xyz(m,k,j,i): mth component to coeffcient of kth paramater (x,y,z)
      integer wyckoff_bns_xyz(3,3,96,27,magcount)
* wyckoff_bns_mag(m,k,j,i): mth component to coeffcient of kth magnetic
* paramater (mx,my,mz)
      integer wyckoff_bns_mag(3,3,96,27,magcount)

* for OG setting (for type-4 groups)
* lattice_og_vectors_count(i): number of lattice vectors defining the lattice
      integer lattice_og_vectors_count(magcount)
* lattice_og_vectors(k,j,i): kth component of the jth lattice vector
* lattice_og_vectors_denom(j,i): common denominator
      integer lattice_og_vectors(3,6,magcount)
      integer lattice_og_vectors_denom(6,magcount)
* for jth operator
* ops_og_point_op(j,i): point operator part
      integer ops_og_point_op(96,magcount)
* ops_og_trans(k,j,i): kth component of translation part
* ops_og_trans_denom(j,i): common denominator
      integer ops_og_trans(3,96,magcount)
      integer ops_og_trans_denom(96,magcount)
* ops_og_timeinv(j,i): 1=no time inversion, -1=time inversion
      integer ops_og_timeinv(96,magcount)
* for jth wyckoff site
* wyckoff_og_fract(k,j,i): kth component of fractional part of wyckoff position
* wyckoff_og_fract_denom(j,i): common denominator
      integer wyckoff_og_fract(3,96,27,magcount)
      integer wyckoff_og_fract_denom(96,27,magcount)
* wyckoff_og_xyz(m,k,j,i): mth component to coeffcient of kth paramater (x,y,z)
      integer wyckoff_og_xyz(3,3,96,27,magcount)
* wyckoff_og_mag(m,k,j,i): mth component to coeffcient of kth magnetic
* paramater (mx,my,mz)
      integer wyckoff_og_mag(3,3,96,27,magcount)

******************************************************************************
* open data file
      open(30,file='magnetic_data.txt')
* read nonhexangonal point operators
      do i=1,48
        read(30,*)n,point_op_label(i),point_op_xyz(i),
     $       ((point_op_matrix(k,j,i),j=1,3),k=1,3)
        if(n.ne.i)stop
     $       'error in numbering of nonhexagonal point operators'
      enddo
* read hexangonal point operators
      do i=1,24
        read(30,*)n,point_op_hex_label(i),
     $       point_op_hex_xyz(i),
     $       ((point_op_hex_matrix(k,j,i),j=1,3),k=1,3)
        if(n.ne.i)stop
     $       'error in numbering of hexagonal point operators'
      enddo
* read data for each magnetic space group
      do i=1,1651
        read(30,*)(nlabelparts_bns(j,i),j=1,2),nlabel_bns(i),
     $       spacegroup_label_unified(i),
     $       spacegroup_label_bns(i),(nlabelparts_og(j,i),j=1,3),
     $       nlabel_og(i),spacegroup_label_og(i)
        read(30,*)magtype(i)
        if(magtype(i).eq.4)then
          read(30,*)((bnsog_point_op(j,k,i),j=1,3),k=1,3),
     $         (bnsog_origin(j,i),j=1,3),bnsog_origin_denom(i)
        endif
        read(30,*)ops_count(i)
        read(30,*)(ops_bns_point_op(j,i),(ops_bns_trans(k,j,i),k=1,3),
     $       ops_bns_trans_denom(j,i),ops_bns_timeinv(j,i),
     $       j=1,ops_count(i))
        read(30,*)lattice_bns_vectors_count(i)
        read(30,*)((lattice_bns_vectors(k,j,i),k=1,3),
     $       lattice_bns_vectors_denom(j,i),
     $       j=1,lattice_bns_vectors_count(i))
        read(30,*)wyckoff_site_count(i)
        do j=1,wyckoff_site_count(i)
          read(30,*)wyckoff_pos_count(j,i),wyckoff_mult(j,i),
     $         wyckoff_label(j,i)
          do k=1,wyckoff_pos_count(j,i)
            read(30,*)(wyckoff_bns_fract(m,k,j,i),m=1,3),
     $           wyckoff_bns_fract_denom(k,j,i),
     $           ((wyckoff_bns_xyz(m,n,k,j,i),m=1,3),n=1,3),
     $           ((wyckoff_bns_mag(m,n,k,j,i),m=1,3),n=1,3)
          enddo
        enddo
        if(magtype(i).eq.4)then
        read(30,*)ops_count(i)
        read(30,*)(ops_og_point_op(j,i),(ops_og_trans(k,j,i),k=1,3),
     $       ops_og_trans_denom(j,i),ops_og_timeinv(j,i),
     $       j=1,ops_count(i))
        read(30,*)lattice_og_vectors_count(i)
        read(30,*)((lattice_og_vectors(k,j,i),k=1,3),
     $       lattice_og_vectors_denom(j,i),
     $       j=1,lattice_og_vectors_count(i))
        read(30,*)wyckoff_site_count(i)
        do j=1,wyckoff_site_count(i)
          read(30,*)wyckoff_pos_count(j,i),wyckoff_mult(j,i),
     $         wyckoff_label(j,i)
          do k=1,wyckoff_pos_count(j,i)
            read(30,*)(wyckoff_og_fract(m,k,j,i),m=1,3),
     $           wyckoff_og_fract_denom(k,j,i),
     $           ((wyckoff_og_xyz(m,n,k,j,i),m=1,3),n=1,3),
     $           ((wyckoff_og_mag(m,n,k,j,i),m=1,3),n=1,3)
          enddo
        enddo
        endif
      enddo
* close data file
      close(30)
      ! filename = 'example.txt'
      ! print *, ops_bns_point_op(:,970)
      ! print *,ops_og_point_op
      ! integer :: iounit = 10
      open(unit=10, file='example2.txt')
      do j = 1, 1651
      !       ! Write each row (96 integers) from the current column to the file, on a new line
            write(10, '(96I3)') (ops_og_point_op(i, j), i = 1, 96)
      end do
      ! ! write(10, *) ops_bns_point_op
      ! close(10)
      end
