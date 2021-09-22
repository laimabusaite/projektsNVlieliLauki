# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
#11111

from NVcenter import *
import matplotlib.pyplot as plt


def nv_ham():
    D = 2870
    Mz = 0
    nv = NVcenter(D, Mz)

    Brange = np.arange(0, 1500., 1.)
    eigenvalues_array = np.zeros((len(Brange), 3))
    eigenvectors_array = np.zeros((len(Brange), 3, 3))

    eigenvalues_array_2 = np.zeros((len(Brange), 3))
    eigenvectors_array_2 = np.zeros((len(Brange), 3, 3))

    for idx, B in enumerate(Brange):
        B_lab0 = np.array([10, 100, B])
        nv.setMagnetic(*B_lab0)

        nv.calculateHamiltonian()
        eigenvalues_array[idx] = nv.eVals[:]
        eigenvectors_array[idx, :, :] = np.abs(nv.eVecs[:, :])

        # eigenvalues_array_2[idx] = np.dot(np.abs(nv.eVecs).transpose().conjugate(), np.dot(nv.eVals, np.abs(nv.eVecs)))
        eigenvalues_array_2[idx] = np.dot(np.abs(nv.eVecs), nv.eVals)
        eigenvectors_array_2[idx, :, :] = np.dot(nv.eVecs, nv.eVecs)

        # print('Hamiltonian:')
        # print(nv.H)
        # print('Eigenvalues:')
        # print(nv.eVals)
        # print('Eigenvectors')
        # print(nv.eVecs)

    plt.figure('Eigenvalues')
    plt.plot(Brange, eigenvalues_array)
    # plt.plot(Brange, eigenvalues_array_2)

    plt.figure('Eigenvectors')
    for i in range(3):
        plt.subplot(3,1,i+1)
        plt.plot(Brange, eigenvectors_array[:,:,i])
        # plt.plot(Brange, eigenvectors_array_2[:, :, i])



    plt.show()



if __name__ == '__main__':
    nv_ham()

